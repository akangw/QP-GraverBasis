import time
import argparse
from helper import *
import os
import torch
from loguru import logger
# os.environ['GRB_LICENSE_FILE'] = '/home/lwb/gurobi1002/gurobi.lic'
fix_seed(0)

def inference(INS_NAME, NUM_INIT_SOLS, NUM_GB, AUGMENT_BATCH=1, SIFT=4):
    t = time.perf_counter()
    ins = os.path.basename(INS_NAME)
    IDX = ins.split('_')[-1]
    SAVE_DIR = f'./results/QPLIB/maple/{NUM_INIT_SOLS}_init_{NUM_GB}_GB'
    SAVE_PATH = f'./results/QPLIB/maple/{NUM_INIT_SOLS}_init_{NUM_GB}_GB/results_{IDX}.txt'
    if os.path.exists(SAVE_PATH):
        return 0
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    TORCH_DTYPE = torch.float
    TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A, b, Q, x_raw_features = parse_lp(INS_NAME, TORCH_DTYPE)
    M = torch.cat((A, b.unsqueeze(-1)), dim=-1)
    M = maximal_linearly_independent_rows(M)
    A = M[:, :-1]
    b = M[:, -1]
    m, n = A.shape
    parse_time = time.perf_counter() - t
    print(f'Parsing time: {parse_time:}')

    # HNF
    t = time.perf_counter()
    HNF, C = column_style_hermite_normal_form(A.cpu().numpy())
    C = torch.FloatTensor(C).cuda()
    B = C[:, m:]
    H = torch.linalg.inv(B.t() @ B) @ B.t()
    Proj_matrix = B @ H
    HNF_time = time.perf_counter() - t
    print('HNF_time: ', HNF_time)
    sparsity = torch.max(torch.norm(B, dim=0, p=1))

    # Collect Graver Basis
    t = time.perf_counter()
    GB = torch.tensor([], dtype=TORCH_DTYPE, device=TORCH_DEVICE)
    x = 2 * torch.rand((NUM_GB, n), dtype=torch.float, device='cuda') - 1
    z = torch.einsum('dn,bn->bd', H, x)
    z = z.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=0.003)
    for epoch in range(1000):
        x = torch.einsum('nd,kd->kn', B, z)
        loss = ((torch.ceil(z) - z) * (z - torch.floor(z))).sum() + 0.85 * torch.norm(x, p=1,
                                                                                       dim=-1).sum() + (
                   (1 / torch.norm(x, p=float('inf'), dim=-1) - 1).relu()).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 and epoch > 100:
            int_z = torch.round(z)
            int_z = torch.unique(int_z, dim=0)
            int_x = torch.einsum('nd,kd->kn', B, int_z)
            varbounds_idx = torch.norm(int_x, p=float('inf'), dim=-1) <= 1
            GB = torch.cat([GB, int_x[varbounds_idx]], dim=0)
            GB = torch.unique(GB, dim=0)
    # Sift the dummy elements with SIFT being a hyper-parameter
    GB = GB[torch.norm(GB, p=1, dim=1) <= sparsity + SIFT]
    zero = torch.zeros(A.shape[1], device=TORCH_DEVICE, dtype=TORCH_DTYPE)
    GB = torch.cat((GB, zero.unsqueeze(0)), dim=0)
    OPTZ_time = time.perf_counter() - t
    print('OPT Z', OPTZ_time)

    # Collect Initial Solutions
    t = time.perf_counter()
    if IDX[0] == '7':
        lr_init = 0.5
        penalty_init = 0.001
    else:
        lr_init = 1
        penalty_init = 0.1
    if NUM_INIT_SOLS > 0:
        initial_xs = get_init_solutions_2(A, b, lr_init, penalty_init, NUM_INIT_SOLS).type(TORCH_DTYPE)
        initial_xs = torch.unique(initial_xs, dim=0)
    init_time = time.perf_counter() - t
    print(torch.unique(initial_xs, dim=0).shape)
    print(f'Initial solutions time:{init_time}')

    # The augmentation step
    GB = GB.clone().detach()
    xs_split = torch.split(initial_xs, int(initial_xs.shape[0]/AUGMENT_BATCH))
    incumbent = float('inf')
    t = time.perf_counter()
    obj_per_sample = []
    for xs in xs_split:
        while True:
            xs_ = xs.clone().detach()
            obj_ = 0.5 * torch.einsum('kn,kn->k', xs_ @ Q, xs_)
            xs = xs.unsqueeze(1) + GB.unsqueeze(0)
            objs = 0.5 * torch.einsum('kln,kln->kl', xs @ Q, xs)
            infeasible_indices = torch.nonzero(torch.any(xs >= 2, dim=-1) | torch.any(xs <= -1, dim=-1)).squeeze(-1)
            objs[infeasible_indices[:, 0], infeasible_indices[:, 1]] = float('inf')
            obj, idx = torch.min(objs, dim=-1)
            xs = xs[torch.arange(xs_.shape[0]), idx, :]

            if torch.allclose(xs, xs_) or torch.allclose(obj, obj_):
                obj_per_sample.append(obj)
                break
    obj_per_sample = torch.cat(obj_per_sample)

    search_time = time.perf_counter() - t
    with open(SAVE_PATH, 'w') as f:
        f.write('OBJ:'+str(torch.min(obj_per_sample).item())+'\n')
        f.write(f'TIME\n init:{init_time}\nHNF:{HNF_time}\nGB:{OPTZ_time}\nsearch:{search_time}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', type=int, default=50)
    parser.add_argument('--gb', type=int, default=100000)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()

    #gp.setParam('LogToConsole', 0)

    for file in os.listdir('./instance/QPLIB'):
        ins_name = os.path.join('./instance/QPLIB', file)
        m = gp.read(ins_name)
        print(ins_name)
        inference(ins_name, args.init, args.gb, args.batch)




