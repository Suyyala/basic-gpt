# simple feed forward neural network for attention block

import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.multiplier = 4
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.multiplier * embed_dim),
            nn.ReLU(),
        )
        # add projection to embedding dimension
        self.proj = nn.Linear(self.multiplier * embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: B, T, C
        B, T, C = x.shape
        # pass through feed forward network
        x = self.ffn(x)
        # project back to embedding dimension
        x = self.proj(x)
        # dropout
        x = self.dropout(x)
        return x