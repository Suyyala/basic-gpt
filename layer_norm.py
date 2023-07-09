# implementation of layer normalization

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm:
    def __init__(self, n_dim, eps=1e-6):
        self.gamma = nn.Parameter(torch.ones(n_dim))
        self.beta = nn.Parameter(torch.zeros(n_dim))
        self.eps = eps

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x: B, T, C
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        xhat = (x - mean) / (std + self.eps)
        return self.gamma * xhat + self.beta
    

# test layer norm
if __name__ == '__main__':
    model = LayerNorm(n_dim=16)
    x = torch.randn(3, 5, 16)
    print(x.shape)
    print(x[0,0])
    out = model(x)
    print(f'mean {out.mean(dim=-1)}')
    print(f'std {out.std(dim=-1)}')
