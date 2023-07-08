# transformer decoder
import torch
import torch.nn as nn
from torch.nn import functional as F
from multi_head import MultiHeadAttention

# extension of bigram model to include attention and positional encoding
class DecoderModel(nn.Module):
    def __init__(self, vocab_size, head_size=16, block_size=128, embed_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.head_size = head_size
        self.block_size = block_size
        self.n_embd = embed_dim
        self.num_heads = 4
        self.token_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.position_embed = nn.Embedding(num_embeddings=block_size, embedding_dim=embed_dim)
        self.attn_head = MultiHeadAttention(num_heads=self.num_heads, embed_dim=embed_dim, head_size=head_size // self.num_heads) # B, T, H
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        # idx: B, T
        # targets: B, T
        B, T = idx.shape
        token_emb = self.token_embed(idx)  # B, T, C
        pos_emb = self.position_embed(torch.arange(T, device=idx.device)) # T, C
        x = token_emb + pos_emb # B, T, C
        x = self.attn_head(x)
        logits = self.out(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_tokens):
        # B, T is indices of current context
        for i in range(max_tokens):
            #print(idx.shape)
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            # get latest timestep logits
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            id_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, id_next), dim=1)
        print(idx.shape)
        return idx

# test decoder
if __name__ == '__main__':
    model = DecoderModel(vocab_size=10)
    idx = torch.randint(0, 10, (3, 5))
    logits, loss = model(idx, targets=idx)
    print(logits.shape)
    print(loss)
    idx = torch.randint(0, 10, (3, 5))
    idx = model.generate(idx, max_tokens=10)
    print(idx.shape)
    print(idx)

