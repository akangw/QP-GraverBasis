import time
import numpy as np
from hsnf import column_style_hermite_normal_form, row_style_hermite_normal_form, smith_normal_form
import numpy as np
import pyscipopt as scp
import torch
import random
from loguru import logger
import gurobipy as gp
# import dgl
# from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F


def maximal_linearly_independent_rows(A):
    """
    使用 PyTorch 求取矩阵 A 的极大线性无关行组。
    """
    # 进行奇异值分解 (SVD)
    U, S, V = torch.svd(A)

    # 获取矩阵的秩
    rank = torch.linalg.matrix_rank(A)

    # 找到前 rank 行的索引
    independent_rows_indices = torch.argsort(S, descending=True)[:rank]

    # 提取极大线性无关行组
    return A[independent_rows_indices]


def parse_lp(ins_name, torch_dtype=torch.float32, device='cuda'):
    m = gp.read(ins_name)
    mvars = m.getVars()
    mcontrs = m.getConstrs()
    Q = torch.zeros(len(mvars), len(mvars), device=device, dtype=torch_dtype)
    obj = m.getObjective()

    obj_linear = obj.getLinExpr()
    A = torch.zeros(len(mcontrs), len(mvars), device=device, dtype=torch_dtype)
    b = torch.zeros(len(mcontrs), device=device, dtype=torch_dtype)
    for c_idx, constr in enumerate(mcontrs):
        b[c_idx] = constr.RHS
        constr = m.getRow(constr)
        for linear_idx in range(constr.size()):
            v_idx = constr.getVar(linear_idx).index
            A[c_idx, v_idx] += int(constr.getCoeff(linear_idx))
    for quad_idx in range(obj.size()):
        i, j = obj.getVar1(quad_idx).index, obj.getVar2(quad_idx).index
        if i == j:
            Q[i, j] += 2 * obj.getCoeff(quad_idx)
        else:
            Q[i, j] += obj.getCoeff(quad_idx)
            Q[j, i] += obj.getCoeff(quad_idx)
    for lin_idx in range(obj_linear.size()):
        i = obj_linear.getVar(lin_idx).index
        Q[i, i] += 2 * obj_linear.getCoeff(lin_idx)

    x_raw_features = torch.empty((len(mvars), 2), dtype=torch_dtype, device=device)
    for i, v in enumerate(mvars):
        x_raw_features[i, 0] = v.LB
        x_raw_features[i, 1] = v.UB
    return A, b, Q, x_raw_features



def loss_func(p, A, b, penalty, para=None, reg=None):
    eps = 1e-16
    if para is None:
        return 0.5 * torch.norm(torch.matmul(A, p) - b, p=2) ** 2 + penalty * (p - p ** 2).sum()
    elif reg is None:
        return torch.norm(0.5 * torch.norm(torch.einsum('mn,kn->km', A, p) - b, p=2, dim=-1) ** 2 + penalty * torch.sum(
            p - p ** 2, dim=-1), p=2, dim=0)
    elif reg is not None:
        return (0.5 * torch.norm(torch.einsum('mn,kn->km', A, p) - b, p=2, dim=-1) ** 2 + penalty * torch.sum(
            -p * torch.log(p + eps) - (1 - p) * torch.log(1 - p + eps), dim=-1)).sum()



def get_init_solutions_2(A, b, lr, penalty, para, num_epoch=9999):
    # Here we solve Ax=b where x \in {0,1}^n
    torch_dtype = torch.float32
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand((para, A.shape[-1]), dtype=torch_dtype, device=torch_device)
    x = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    for epoch in range(num_epoch):

        Ax = torch.einsum('mn,kn->km', A, x)
        loss = (torch.norm(Ax - b.unsqueeze(0), p=2, dim=-1) ** 2).sum() + penalty * (x - x ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x.data = x.data.clamp(min=0, max=1)
        int_x = torch.round(x).clone().detach()
        if epoch % 100 == 0:
            print(f'epoch{epoch}, {loss.item()},',
                  torch.norm(torch.einsum('mn,kn->km', A, int_x) - b, p=1, dim=-1).mean().item())
            if torch.norm(torch.einsum('mn,kn->km', A, int_x) - b, p=1, dim=-1).mean().item() == 0:
                return int_x
    feasible_indices = torch.nonzero(torch.norm(torch.einsum('mn,kn->km', A, x) - b, p=1, dim=-1) == 0).squeeze(-1)
    return int_x[feasible_indices]






def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


