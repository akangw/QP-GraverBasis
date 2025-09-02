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
    X_L_bar = x_raw_features[:,0]
    X_U_bar = x_raw_features[:,1]
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

    # Collect Initial Solutions
    def get_init_solutions_3(A, b, lr, penalty, num=1, num_epoch=1000):
        # Here we solve Ax=b where x \in {0,1}^m and x \in [X_L_bar, X_U_bar]^(n-m)
        torch_dtype = torch.float32
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n = A.shape[-1]#dim of x
        cont_dim = n - m#dim of continuous variables

        binary_part = torch.rand((num, m), dtype=torch_dtype, device=torch_device)
        cont_part = X_L_bar + (X_U_bar - X_L_bar) * torch.rand((num, cont_dim), dtype=torch_dtype, device=torch_device)
        x=torch.cat((binary_part, cont_part), dim=-1)
        x = x.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([x], lr=lr)

        for epoch in range(num_epoch):
            Ax = torch.einsum('mn,kn->km', A, x)

            #x[num,:m] is binary, x[num,m:] is continuous   
            eq_loss = (torch.norm(Ax - b.unsqueeze(0), p=2, dim=-1) ** 2).sum()  
            binary_loss = penalty * [(x[:,:m] - torch.floor(x[:,:m]))*(torch.ceil(x[:,:m])-x[:,:m])].sum()
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
            
            if epoch % 100 == 0:
                print(f'epoch{epoch}, {total_loss.item()},',
                    torch.norm(torch.einsum('mn,kn->km', A, initial_x) - b, p=1, dim=-1).mean().item())
            if torch.norm(torch.einsum('mn,kn->km', A, initial_x) - b, p=1, dim=-1).mean().item() <= 1e-6:
                break

        return initial_x
    
    #GDA-algorithm
    def GDA(A, B, x0, lambda1_0, lambda2_0, lambda3_0, X_L_bar, X_U_bar, epsilon=1e-6, alpha=1e-3, beta=1e-3, T=1000):
        '''
        A: m*n matrix
        B: n*(n-m) matrix, basis of null space of A
        x0: n*1 vector
        z0: (n-m)*1 vector
        lambda1_0: m*1 vector
        lambda2_0: (n-m)*1 vector
        lambda3_0: (n-m)*1 vector
        x_row_features[:,0]:X_L_bar, n*1 vector
        x_row_features[:,1]:X_U_bar, n*1 vector
        epsilon: limit for stopping criterion
        alpha: learing rate for lambda update
        beta: learing rate for z update
        T: max iteration
        return: z, lambda1, lambda2, lambda3
        '''
        # Objective function
        #Unsured about the sign here
        def f(x):
            return 1/2 * x.T @ Q @ x


        def Phi(lambda1,lambda2, lambda3,z):
            term1 = f(x0 + B@z)

            fractional_terms = [(zi - np.floor(zi))@(np.ceil(zi) - zi) for zi in z]
            term2 = lambda1 @ torch.sum(torch.tensor(fractional_terms, device=z.device))

            Bz = B @ z
            lower_terms = (Bz - X_L_bar).relu()
            term3 = lambda2 @ lower_terms

            upper_terms = (X_U_bar - Bz).relu()
            term4 = lambda3 @ upper_terms

            return term1 + term2 + term3 + term4
        
        def Gradient_lambda(lambda1,lambda2, lambda3,z):
            
            nabla_lambda1 = torch.sum(torch.tensor([(zi - np.floor(zi))@(np.ceil(zi) - zi) for zi in z], device=z.device))
            nabla_lambda2 = (B @ z - X_L_bar).relu()
            nabla_lambda3 = (X_U_bar - B @ z).relu()

            return nabla_lambda1, nabla_lambda2, nabla_lambda3 
        
        def subgradient_z_analytic(z, lambda1, lambda2, lambda3, x_bar, B, X_L_bar, X_U_bar):

            Bz = B @ z
            y = f(x_bar+Bz)
            grad = torch.autograd.grad(y, z, retain_graph=True)[0]
            natala_z = B@grad + lambda1@torch.tensor([torch.ceil(zi)+torch.floor(zi)-2*zi for zi in z], device=z.device)\
                    + lambda2@(B.T@(Bz - X_L_bar).relu()) - lambda3@(B.T@(X_U_bar - Bz).relu())
            return natala_z
        
        # Initialization
        TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z = torch.zeros((B.shape[1],1), device=TORCH_DEVICE, requires_grad=True)
        lambda1 = torch.tensor(lambda1_0, device=TORCH_DEVICE, requires_grad=False)
        lambda2 = torch.tensor(lambda2_0, device=TORCH_DEVICE, requires_grad=False)
        lambda3 = torch.tensor(lambda3_0, device=TORCH_DEVICE, requires_grad=False)

        # Main loop
        for t in range(T):
            # Update z
            natala_z = subgradient_z_analytic(z, lambda1, lambda2, lambda3, x0, B, X_L_bar, X_U_bar)
            z = z - beta * natala_z
            z = z.detach().requires_grad_(True)
            # Update lambda
            natala_lambda1, natala_lambda2, natala_lambda3 = Gradient_lambda(lambda1, lambda2, lambda3, z)
            lambda1 = (lambda1 + alpha * natala_lambda1)
            lambda2 = (lambda2 + alpha * natala_lambda2)
            lambda3 = (lambda3 + alpha * natala_lambda3)

            # Stopping criterion
            if torch.linalg.norm(natala_z) < epsilon and torch.linalg.norm(natala_lambda2) < epsilon and torch.linalg.norm(natala_lambda3) < epsilon:
                break

        return z, lambda1, lambda2, lambda3
    
    # Collect Initial Solutions
    t = time.perf_counter()
    if IDX[0] == '7':
        lr_init = 0.5
        penalty_init = 0.001
    else:
        lr_init = 1
        penalty_init = 0.1
    if NUM_INIT_SOLS > 0:
        x0 = get_init_solutions_3(A, b, lr_init, penalty_init, NUM_INIT_SOLS).type(TORCH_DTYPE)
    init_time = time.perf_counter() - t
    print(f'Initial solutions time:{init_time}')

    # GDA
    lambda1_0 =torch.ones(m)*1e-3
    lambda2_0 =torch.ones(n - m)*1e-3
    lambda3_0 =torch.ones(n - m)*1e-3
    GDA(A, B, x0[0].unsqueeze(-1), lambda1_0, lambda2_0, lambda3_0, X_L_bar, X_U_bar, epsilon=1e-6, alpha=1e-3, beta=1e-3, T=1000)