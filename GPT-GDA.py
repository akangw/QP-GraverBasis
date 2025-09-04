import time
import argparse
from helper import *
import os
import torch
import matplotlib.pyplot as plt
from loguru import logger
import gurobipy as gp

# os.environ['GRB_LICENSE_FILE'] = '/home/lwb/gurobi1002/gurobi.lic'
fix_seed(0)


def inference(INS_NAME, NUM_INIT_SOLS, AUGMENT_BATCH=1, SIFT=4):
    t0 = time.perf_counter()
    ins = os.path.basename(INS_NAME)
    IDX = ins.split('_')[-1]
    SAVE_DIR = f'./results/QPLIB/maple/{NUM_INIT_SOLS}_init'
    SAVE_PATH = f'./results/QPLIB/maple/{NUM_INIT_SOLS}_init/results_{IDX}.txt'
    if os.path.exists(SAVE_PATH):
        return 0
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    TORCH_DTYPE = torch.float32
    TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parse_lp should return CPU tensors; move them to the chosen device
    A, b, Q, x_raw_features = parse_lp(INS_NAME, TORCH_DTYPE)
    A = A.to(TORCH_DEVICE)
    b = b.to(TORCH_DEVICE)
    Q = Q.to(TORCH_DEVICE)
    x_raw_features = x_raw_features.to(TORCH_DEVICE)

    X_L_bar = x_raw_features[:, 0]
    X_U_bar = x_raw_features[:, 1]
    M = torch.cat((A, b.unsqueeze(-1)), dim=-1)
    M = maximal_linearly_independent_rows(M)
    A = M[:, :-1]
    b = M[:, -1]
    m, n = A.shape
    parse_time = time.perf_counter() - t0
    print(f'Parsing time: {parse_time:}')

    # HNF
    t1 = time.perf_counter()
    HNF, C = column_style_hermite_normal_form(A.cpu().numpy())
    # move C to the same device as torch
    C = torch.FloatTensor(C).to(TORCH_DEVICE)
    B = C[:, m:]

    # Make B nicer: compute pseudo-inverse style H (as in your code) and projection (kept for compatibility)
    H = torch.linalg.inv(B.t() @ B) @ B.t()
    Proj_matrix = B @ H

    HNF_time = time.perf_counter() - t1
    print('HNF_time: ', HNF_time)
    sparsity = torch.max(torch.norm(B, dim=0, p=1))

    # === Minimal fixes applied globally ===
    # 1) Symmetrize Q to get stable gradient
    Qsym = 0.5 * (Q + Q.T)
    # small regularization to stabilize if needed
    Qsym = Qsym + 1e-12 * torch.eye(Qsym.shape[0], device=TORCH_DEVICE)

    # 2) Orthonormalize B's columns (QR) to avoid large operator norm
    # Note: QR on CUDA is supported; ensure B is float
    B = B.to(TORCH_DEVICE)
    try:
        Qb, _ = torch.linalg.qr(B, mode='reduced')
        B = Qb
    except Exception:
        # fallback: normalize columns
        B = B / (torch.linalg.norm(B, dim=0, keepdim=True) + 1e-12)

    # Collect Initial Solutions
    def get_init_solutions_3(A, b, lr, penalty, num=1, num_epoch=1000):
        # Here we solve Ax=b where x \in {0,1}^m and x \in [X_L_bar, X_U_bar]^(n-m)
        n_local = A.shape[-1]  # dim of x
        cont_dim = n_local - m  # dim of continuous variables

        binary_part = torch.rand((num, m), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        cont_part = X_L_bar[m:] + (X_U_bar[m:] - X_L_bar[m:]) * torch.rand((num, cont_dim), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        x = torch.cat((binary_part, cont_part), dim=-1)
        x = x.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([x], lr=lr)

        for epoch in range(num_epoch):
            Ax = torch.einsum('mn,kn->km', A, x)

            # x[num,:m] is binary, x[num,m:] is continuous
            eq_loss = (torch.norm(Ax - b.unsqueeze(0), p=2, dim=-1) ** 2).sum()
            binary_loss = penalty * ((x[:, :m] - torch.floor(x[:, :m])) * (torch.ceil(x[:, :m]) - x[:, :m])).sum()
            total_loss = eq_loss + binary_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # projection
            with torch.no_grad():
                x[:, :m] = x[:, :m].clamp(min=0, max=1)
                x[:, m:] = x[:, m:].clamp(min=X_L_bar[m:], max=X_U_bar[m:])
            initial_x = x.clone().detach()
            initial_x[:, :m] = torch.round(initial_x[:, :m])  # round the binary part

            if torch.norm(torch.einsum('mn,kn->km', A, initial_x) - b, p=1, dim=-1).mean().item() <= 1e-6:
                break

        return initial_x.squeeze(0)

    # GDA-algorithm (modified and integrated)
    def GDA(A, B, x0, lambda1_0, lambda2_0, lambda3_0, X_L_bar, X_U_bar,
            epsilon=1e-6, alpha=1e-6, beta=1e-6, T=10000,
            GRAD_CLIP=1e6, L_MAX=1e6, PROJECT_X=True):
        '''
        Modified GDA with:
         - explicit term0 = B^T Qsym (x0 + Bz)
         - B orthonormalized outside
         - optional x projection (PROJECT_X)
         - diagnostics and plotting
        '''
        history = {'t': [], 'term0': [], 'term1': [], 'term2': [], 'term3': []}

        def f(x):
            return 0.5 * x @ Qsym @ x

        def subgradient_z_analytic(z, lambda1, lambda2, lambda3, x_bar, B, X_L_bar, X_U_bar):
            Bz = B @ z
            x = x_bar + Bz

            # explicit term0
            term0 = B.T @ (Qsym @ x)

            # integerity smoothing term (still using ceil/floor expression you had)
            term1 = lambda1 * (torch.ceil(z) + torch.floor(z) - 2 * z)

            term2 = lambda2 * (B.T @ (Bz - X_L_bar).relu())
            term3 = lambda3 * (B.T @ (X_U_bar - Bz).relu())

            natala_z = term0 + term1 + term2 + term3
            return natala_z, term0, term1, term2, term3

        # initialize primal and dual variables on device
        z = torch.zeros(B.shape[1], device=TORCH_DEVICE)
        lambda1 = lambda1_0.clone().detach().to(TORCH_DEVICE)
        lambda2 = lambda2_0.clone().detach().to(TORCH_DEVICE)
        lambda3 = lambda3_0.clone().detach().to(TORCH_DEVICE)

        for t in range(T):
            natala_z, term0, term1, term2, term3 = subgradient_z_analytic(
                z, lambda1, lambda2, lambda3, x0, B, X_L_bar, X_U_bar
            )

            # diagnose
            if t % 10 == 0:
                with torch.no_grad():
                    x = x0 + B @ z
                    # compute small projected matrix spectrum
                    try:
                        S = (B.T @ Qsym @ B).to(torch.float64)
                        eigs = torch.linalg.eigvalsh(S).cpu().numpy()
                        max_eig = float(eigs.max())
                    except Exception:
                        max_eig = float(torch.linalg.svdvals(B.T @ Qsym @ B)[0].item())

                    print(f"t={t} | ||term0||={term0.norm().item():.3e} | ||x||={x.norm().item():.3e} | BtQB_maxeig={max_eig:.3e}")

                history['t'].append(t)
                history['term0'].append(term0.detach().cpu().numpy().mean())
                history['term1'].append(term1.detach().cpu().numpy().mean())
                history['term2'].append(term2.detach().cpu().numpy().mean())
                history['term3'].append(term3.detach().cpu().numpy().mean())

            # gradient clipping on primal gradient
            natala_z_norm = torch.linalg.norm(natala_z)
            if torch.isfinite(natala_z_norm) and natala_z_norm > GRAD_CLIP:
                natala_z = natala_z * (GRAD_CLIP / natala_z_norm)

            # primal update
            z = z - beta * natala_z

            # optional: project x back into box and map back to z to prevent x runaway
            if PROJECT_X:
                with torch.no_grad():
                    x = x0 + B @ z
                    # enforce variable bounds
                    x[m:] = x[m:].clamp(min=X_L_bar[m:], max=X_U_bar[m:])
                    x[:m] = x[:m].clamp(0.0, 1.0)
                    # map back to z-coordinates (since B columns are orthonormal)
                    z = B.T @ (x - x0)

            # dual updates (ascent)
            nabla_lambda1 = ((z - torch.floor(z)) * (torch.ceil(z) - z)).sum()
            nabla_lambda2 = (B @ z - X_L_bar).relu().sum()
            nabla_lambda3 = (X_U_bar - B @ z).relu().sum()

            lambda1 = (lambda1 + alpha * nabla_lambda1)
            lambda2 = (lambda2 + alpha * nabla_lambda2)
            lambda3 = (lambda3 + alpha * nabla_lambda3)

            # lambda1 = (lambda1 + alpha * nabla_lambda1).clamp(min=0.0, max=L_MAX)
            # lambda2 = (lambda2 + alpha * nabla_lambda2).clamp(min=0.0, max=L_MAX)
            # lambda3 = (lambda3 + alpha * nabla_lambda3).clamp(min=0.0, max=L_MAX)

            # simple stopping criteria
            if torch.linalg.norm(natala_z) < epsilon and nabla_lambda2.abs() < epsilon and nabla_lambda3.abs() < epsilon:
                print(f"Stopped at iter {t} by residuals")
                break

            # break early for debugging if requested
            # if t > 2000:
            #     print('Breaking early for safety (t>2000)')
            #     break

        print(f'Converged at iteration {t}')

        # final plot of stored history
        if len(history['t']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(history['t'], history['term0'], label='term0', marker='o')
            plt.plot(history['t'], history['term1'], label='term1', marker='s')
            plt.plot(history['t'], history['term2'], label='term2', marker='^')
            plt.plot(history['t'], history['term3'], label='term3', marker='d')
            plt.xlabel('Iteration (t)')
            plt.ylabel('Value')
            plt.title('Complete Terms Evolution')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            fname = f"terms_plot_t_{t}.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Saved terms plot to {fname}")
            plt.close()

        return z, lambda1, lambda2, lambda3

    # Collect Initial Solutions
    t2 = time.perf_counter()
    if IDX[0] == '7':
        lr_init = 0.5
        penalty_init = 0.001
    else:
        lr_init = 1
        penalty_init = 0.1
    if NUM_INIT_SOLS > 0:
        x0 = get_init_solutions_3(A, b, lr_init, penalty_init, NUM_INIT_SOLS).type(TORCH_DTYPE).to(TORCH_DEVICE)
    else:
        # fallback to a feasible-looking x0 (projected zero)
        x0 = torch.zeros(n, device=TORCH_DEVICE, dtype=TORCH_DTYPE)
        # try to set continuous part to mid-box
        x0[m:] = (X_L_bar[m:] + X_U_bar[m:]) / 2.0

    init_time = time.perf_counter() - t2
    print(f'Initial solutions time:{init_time}')
    print(f'initial solution:{x0}')

    # GDA
    lambda1_0 = torch.tensor(1e-3, dtype=TORCH_DTYPE)
    lambda2_0 = torch.tensor(1e-3, dtype=TORCH_DTYPE)
    lambda3_0 = torch.tensor(1e-3, dtype=TORCH_DTYPE)
    t3 = time.perf_counter()

    z_opt, lambda1_opt, lambda2_opt, lambda3_opt = GDA(
        A, B, x0, lambda1_0, lambda2_0, lambda3_0, X_L_bar, X_U_bar,
        epsilon=1e-6, alpha=1e-6, beta=1e-6, T=10000,
        GRAD_CLIP=1e6, L_MAX=1e6, PROJECT_X=True
    )

    GDA_time = time.perf_counter() - t3
    print(f'GDA time:{GDA_time}')
    print(f'z:{z_opt}, lambda1: {lambda1_opt.item()}, lambda2: {lambda2_opt.item()}, lambda3: {lambda3_opt.item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_init_sols', type=int, default=1)
    args = parser.parse_args()

    gp.setParam('OutputFlag', 0)

    for i, file in enumerate(os.listdir('/home/opt/data/szy/MAPLE/instance/QPLIB')):
        if i >= 1:
            break

        ins_name = os.path.join('/home/opt/data/szy/MAPLE/instance/QPLIB', file)
        m = gp.read(ins_name)
        print(ins_name)
        inference(ins_name, args.num_init_sols)
        print('==================================================')
