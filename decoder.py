# transformer decoder
import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import AttnHead

# extension of bigram model to include attention and positional encoding
class DecoderModel(nn.Module):
    def __init__(self, vocab_size, head_size=16):
        super().__init__()
        self.token_embeddings_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)
        self.attn_head = AttnHead(embed_dim=vocab_size, head_size=head_size)
        self.pos_enc = nn.Parameter(torch.zeros(1, 1, vocab_size))
        self.out = nn.Linear(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: B, T
        # targets: B, T
        logits = self.token_embeddings_table(idx)
        logits = logits + self.pos_enc
        logits = self.attn_head(logits)
        logits = self.out(logits)
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
            logits, loss = self(idx)
            # get latest timestep logits
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            id_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, id_next), dim=1)
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

