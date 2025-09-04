import time
import argparse
from helper import *
import os
import torch
import matplotlib.pyplot as plt
from loguru import logger
import torch.func as func
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
    A, b, Q, x_raw_features = parse_lp(INS_NAME, TORCH_DTYPE)
    X_L_bar = x_raw_features[:,0]
    X_U_bar = x_raw_features[:,1]
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
    C = torch.FloatTensor(C).cuda()
    B = C[:, m:]
    H = torch.linalg.inv(B.t() @ B) @ B.t()
    Proj_matrix = B @ H
    HNF_time = time.perf_counter() - t1
    print('HNF_time: ', HNF_time)
    sparsity = torch.max(torch.norm(B, dim=0, p=1))

    # Collect Initial Solutions
    def get_init_solutions_3(A, b, lr, penalty, num=1, num_epoch=1000):
        # Here we solve Ax=b where x \in {0,1}^m and x \in [X_L_bar, X_U_bar]^(n-m)
        n = A.shape[-1]#dim of x
        cont_dim = n - m#dim of continuous variables

        binary_part = torch.rand((num, m), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        cont_part = X_L_bar[m:] + (X_U_bar[m:] - X_L_bar[m:]) * torch.rand((num, cont_dim), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        x=torch.cat((binary_part, cont_part), dim=-1)
        x = x.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([x], lr=lr)

        for epoch in range(num_epoch):
            Ax = torch.einsum('mn,kn->km', A, x)

            #x[num,:m] is binary, x[num,m:] is continuous   
            eq_loss = (torch.norm(Ax - b.unsqueeze(0), p=2, dim=-1) ** 2).sum()  
            binary_loss = penalty * ((x[:,:m] - torch.floor(x[:,:m]))*(torch.ceil(x[:,:m])-x[:,:m])).sum()
            total_loss = eq_loss + binary_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            #projection
            with torch.no_grad():
                x[:,:m] = x[:,:m].clamp(min=0, max=1)
                x[:,m:] = x[:,m:].clamp(min=X_L_bar[m:], max=X_U_bar[m:])
            initial_x = x.clone().detach()
            initial_x[:,:m] = torch.round(initial_x[:,:m])#round the binary part
            
            # if epoch % 100 == 0:
            #     print(f'epoch{epoch}, {total_loss.item()},',
            #         torch.norm(torch.einsum('mn,kn->km', A, initial_x) - b, p=1, dim=-1).mean().item())
            if torch.norm(torch.einsum('mn,kn->km', A, initial_x) - b, p=1, dim=-1).mean().item() <= 1e-6:
                break
        
        return initial_x.squeeze(0)
    
    #GDA-algorithm
    def GDA(A, B, x0, lambda1_0, lambda2_0, lambda3_0, X_L_bar, X_U_bar, epsilon=1e-6, alpha=1e-3, beta=1e-3, T=1000):
        '''
        A: m*n matrix
        B: n*(n-m) matrix, basis of null space of A
        x0: n vector
        z0: (n-m) vector
        lambda1_0: 1 vector
        lambda2_0: 1 vector
        lambda3_0: 1 vector
        x_row_features[:,0]:X_L_bar, n*1 vector
        x_row_features[:,1]:X_U_bar, n*1 vector
        epsilon: limit for stopping criterion
        alpha: learing rate for lambda update
        beta: learing rate for z update
        T: max iteration
        return: z, lambda1, lambda2, lambda3
        '''
        # 存储中间变量的列表
        history = {
            't': [],
            'term0': [],
            'term1': [],
            'term2': [],
            'term3': []
        }
        # Objective function
        #Unsured about the sign here
        def f(x):
            return 1/2 * x @ Q @ x


        def Phi(lambda1,lambda2, lambda3,z):
            term1 = f(x0 + B@z)

            fractional_elems = [(zi - torch.floor(zi)) * (torch.ceil(zi) - zi) for zi in z]
            fractional_sum = torch.tensor(fractional_elems, dtype=TORCH_DTYPE, device=TORCH_DEVICE).sum()
            term2 = lambda1 * fractional_sum

            Bz = B @ z
            lower_elems = [(B[i,:]@z - X_L_bar[i]).relu() for i in range(B.shape[0])]
            lower_terms = torch.tensor(lower_terms, dtype=TORCH_DTYPE, device=TORCH_DEVICE).sum()
            term3 = lambda2 * lower_terms

            upper_elems = [(X_U_bar[i] - B[i,:]@z).relu() for i in range(B.shape[0])]
            upper_terms = torch.tensor(upper_terms, dtype=TORCH_DTYPE, device=TORCH_DEVICE).sum()
            term4 = lambda3 * upper_terms

            return term1 + term2 + term3 + term4
        
        def Gradient_lambda(lambda1,lambda2, lambda3,z):
            
            nabla_lambda1 =  torch.tensor([(zi - torch.floor(zi)) * (torch.ceil(zi) - zi) for zi in z],
                                    dtype=TORCH_DTYPE,device=TORCH_DEVICE).clone().detach().sum() 
            nabla_lambda2 = (B @ z - X_L_bar).relu().to(dtype=TORCH_DTYPE, device=TORCH_DEVICE).clone().detach().sum()
            nabla_lambda3 = (X_U_bar - B @ z).relu().to(dtype=TORCH_DTYPE, device=TORCH_DEVICE).sum()

            return nabla_lambda1, nabla_lambda2, nabla_lambda3 
        
        def subgradient_z_analytic(z, lambda1, lambda2, lambda3, x_bar, B, X_L_bar, X_U_bar):

            Bz = B @ z
            y = f(x_bar+Bz)
            grad = torch.autograd.grad(y, z, retain_graph=True)[0]
            term0 = grad
            # term0 = B.T @ (Q @ (x_bar + Bz))
            term1 = lambda1 * (torch.ceil(z) + torch.floor(z) - 2*z)
            term2 = lambda2 * (B.T @ (Bz - X_L_bar).relu())
            term3 = lambda3 * (B.T @ (X_U_bar - Bz).relu())
            natala_z = term0 + term1 + term2 + term3
            return natala_z, term0, term1, term2, term3
        
        # Initialization
        # Initialize z as a zero vector of length equal to the null space dimension (n-m), used as optimization variable
        z = torch.zeros(B.shape[1], device=TORCH_DEVICE, requires_grad=True)
        lambda1 = lambda1_0.clone().detach().to(TORCH_DEVICE).requires_grad_(True)
        lambda2 = lambda2_0.clone().detach().to(TORCH_DEVICE).requires_grad_(True)
        lambda3 = lambda3_0.clone().detach().to(TORCH_DEVICE).requires_grad_(True)

        # Main loop
        for t in range(T):
            # Update z
            natala_z,term0, term1, term2, term3 = subgradient_z_analytic(z, lambda1, lambda2, lambda3, x0, B, X_L_bar, X_U_bar)

            #调试步骤
            if t % 10 == 0:
                print(f"t: {t}, z: {z}, natala_z: {natala_z}")
                # 存储数据（转换为numpy以便绘图）
                history['t'].append(t)
                history['term0'].append(term0.detach().cpu().numpy().mean())  # 取均值简化绘图
                # history['term1'].append(term1.detach().cpu().numpy().mean())
                # history['term2'].append(term2.detach().cpu().numpy().mean())
                # history['term3'].append(term3.detach().cpu().numpy().mean())

            if t == 116:
                print(f"natala_z: {natala_z}, z: {z},"
                      f"natala_lambda1: {natala_lambda1}, lambda1: {lambda1},"
                      f"natala_lambda2:{natala_lambda2} lambda2: {lambda2},"
                      f"natala_lambda3:{natala_lambda3}, lambda3: {lambda3}")
            # if torch.isnan(natala_z).any() :
            #     raise ValueError(f"迭代 t={t} 出现nan 11111")
            



            
            # # 若发现nan，立即停止并报错
            # if torch.isnan(z).any() or torch.isnan(lambda1).any():
            #     raise ValueError(f"迭代 t={t} 出现nan 22222")
            
            # Update lambda
            natala_lambda1, natala_lambda2, natala_lambda3 = Gradient_lambda(lambda1, lambda2, lambda3, z)

            z = z - beta * natala_z
                       
            lambda1 = (lambda1 + alpha * natala_lambda1)
            lambda2 = (lambda2 + alpha * natala_lambda2)
            lambda3 = (lambda3 + alpha * natala_lambda3)

            # Stopping criterion
            if torch.linalg.norm(natala_z) < epsilon and torch.linalg.norm(natala_lambda2) < epsilon and torch.linalg.norm(natala_lambda3) < epsilon:
                break

            #调试步骤
            if t > 117:
                break
        print(f'Converged at iteration {t}')
        
        # 调试步骤 最后绘制完整历史曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history['t'], history['term0'], label='term0', marker='o')
        # plt.plot(history['t'], history['term1'], label='term1', marker='s')
        # plt.plot(history['t'], history['term2'], label='term2', marker='^')
        # plt.plot(history['t'], history['term3'], label='term3', marker='d')
        
        plt.xlabel('Iteration (t)')
        plt.ylabel('Value')
        plt.title('Complete Terms Evolution')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(f"term0_{t}.png", dpi=300, bbox_inches='tight')
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
        x0 = get_init_solutions_3(A, b, lr_init, penalty_init, NUM_INIT_SOLS).type(TORCH_DTYPE)
    init_time = time.perf_counter() - t2
    print(f'Initial solutions time:{init_time}')
    print(f'initial solution:{x0}')

    # GDA
    lambda1_0 =torch.tensor([1e-3], dtype=TORCH_DTYPE)
    lambda2_0 =torch.tensor([1e-3], dtype=TORCH_DTYPE)
    lambda3_0 =torch.tensor([1e-3], dtype=TORCH_DTYPE)
    t3 = time.perf_counter()
    z_opt, lambda1_opt, lambda2_opt, lambda3_opt =GDA(A, B, x0, lambda1_0, lambda2_0, lambda3_0, X_L_bar, X_U_bar, epsilon=1e-6, alpha=1e-6, beta=1e-6, T=10000)
    GDA_time = time.perf_counter() - t3
    print(f'GDA time:{GDA_time}')
    print(f'z:{z_opt}, lambda1: {lambda1_opt.item()}, lambda2: {lambda2_opt.item()}, lambda3: {lambda3_opt.item()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_init_sols', type=int, default=1)
    args = parser.parse_args()

    gp.setParam('OutputFlag', 0)

    for i, file in enumerate(os.listdir('/home/opt/data/szy/MAPLE/instance/QPLIB')):
        if i >=1:
            break

        ins_name = os.path.join('/home/opt/data/szy/MAPLE/instance/QPLIB', file)
        m=gp.read(ins_name)
        print(ins_name)
        inference(ins_name, args.num_init_sols)
        print('==================================================')
