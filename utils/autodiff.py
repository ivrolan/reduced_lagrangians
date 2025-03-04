import torch


def matrix_vec_diff(M, x):
    # autodiff square matrix M wrt vector x: M is nxn matrix, x is dim m
    bs = M.shape[0]
    n = M.shape[1]
    m = x.shape[1]

    M_vec = M.view(bs, n * n)
    dMdx = torch.zeros(bs, n * n, m, device=M.device)
    ones_grad = torch.ones_like(M_vec[:, 0], device=M.device)

    for i in range(0, n*n):
        dMdx[:, i] = torch.autograd.grad(M_vec[:, i], x, grad_outputs=ones_grad, retain_graph=True,
                                         create_graph=True, allow_unused=True)[0]

    dMdx = dMdx.view(bs, n, n, m)

    return dMdx

