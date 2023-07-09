# self attention head
import torch
import torch.nn as nn
from torch.nn import functional as F

class AttnHead(nn.Module):
    def __init__(self, embed_dim, head_size=16):
        super().__init__()
        self.head_size = head_size
        self.k = nn.Linear(embed_dim, head_size)
        self.v = nn.Linear(embed_dim, head_size)
        self.q = nn.Linear(embed_dim, head_size)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer('mask', torch.tril(torch.ones(head_size, head_size)))

    def forward(self, x):
        B, T, C = x.shape
        key = self.k(x) # B, T, H
        value = self.v(x) # B, T, H
        query = self.q(x) # B, T, H
        # self attention
        w = query @ key.transpose(-2, -1) * self.head_size ** -0.5 # B, T, T
        # mask out future tokens
        w = w.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        # normalize
        w = F.softmax(w, dim=-1)
        # dropout
        w = self.dropout(w)
        # weighted sum
        out = w @ value # B, T, T @ B, T, H -> B, T, H
        # print(out.shape)
        return out
