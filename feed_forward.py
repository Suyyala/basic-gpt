# simple feed forward neural network for attention block

import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.ffn(x)