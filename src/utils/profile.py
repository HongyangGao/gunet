import torch
from torch import nn
from pytorch_memlab import MemReporter
import time


class LGCN(nn.Module):
    def __init__(self):
        super(LGCN, self).__init__()

    def forward(self, A, F, B, X, W):
        Y = torch.matmul(torch.transpose(B, 0, 1), X)  # E x N x C
        Y = torch.matmul(F, Y)                         # E x E x C
        X2 = torch.matmul(B, Y)                        # E x N x C
        X1 = torch.matmul(A, X)                        # N x N x C
        X = X1 + X2                                   
        X = torch.matmul(X, W)                         # N x C x C
        return X


class WLGCN(nn.Module):
    def __init__(self):
        super(WLGCN, self).__init__()

    def forward(self, A, B, X, W):
        H = torch.matmul(B, torch.transpose(B, 0, 1))  # N x N x E
        F = torch.matmul(H, H)                         # N x N x N
        X2 = torch.matmul(F, X)                        # N x N x C
        X1 = torch.matmul(A, X)                        # N x N x C
        X = X1 + X2
        X = torch.matmul(X, W)                         # N x C x C
        return X


if __name__ == '__main__':
    N = 2000
    E = 150000
    C = 64

    func_name = 'WLGCN'

    start = time.time()

    if func_name == 'WLGCN':
        A = torch.ones([N, N], dtype=torch.float32)
        X = torch.ones([N, C], dtype=torch.float32)
        W = torch.ones([C, C], dtype=torch.float32)
        # F = torch.ones([E, E], dtype=torch.float32)
        B = torch.ones([N, E], dtype=torch.float32)

        func = WLGCN()
        outs = func(A, B, X, W)
        reporter = MemReporter()
        reporter.report()
    elif func_name == 'LGCN':
        A = torch.ones([N, N], dtype=torch.float32)
        X = torch.ones([N, C], dtype=torch.float32)
        W = torch.ones([C, C], dtype=torch.float32)
        F = torch.ones([E, E], dtype=torch.float32)
        B = torch.ones([N, E], dtype=torch.float32)

        func = LGCN()
        outs = func(A, F, B, X, W)
        reporter = MemReporter()
        reporter.report()


    print('time---------->', time.time() - start)
