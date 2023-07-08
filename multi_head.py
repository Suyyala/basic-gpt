# Multihead attention from single head attention
import torch
import torch.nn as nn
from torch.nn import functional as F

from attention import AttnHead

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=1, embed_dim=128, head_size=16):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([AttnHead(embed_dim=embed_dim, head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: B, T, C
        B, T, C = x.shape
        # concat all heads
        out = torch.cat([head(x) for head in self.heads], dim=-1) # B, T, H * num_heads
        # project back to embedding dimension
        out = self.proj(out) # B, T, C
        return out
    

# test multihead attention
if __name__ == '__main__':
    model = MultiHeadAttention(num_heads=8)
    x = torch.randn(3, 5, 16)
    out = model(x)
    print(out.shape)
    print(out)