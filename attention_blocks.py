# attention blocks are composed of multihead attention and feed forward neural network


from multi_head import MultiHeadAttention
from feed_forward import FeedForward

import torch
import torch.nn as nn
from torch.nn import functional as F

from multi_head import MultiHeadAttention
from feed_forward import FeedForward

class AttentionBlock(nn.Module):
    def __init__(self, num_heads=1,  embed_dim=128):
        super().__init__()
        self.head_size = embed_dim // num_heads
        self.attn_head = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim, head_size=self.head_size)
        self.ffwd = FeedForward()

    def forward(self, x):
        # x: B, T, C
        B, T, C = x.shape
        # add residual connection
        x = x + self.attn_head(x)
        # add residual connection
        x = x + self.ffwd(x)
        return x
