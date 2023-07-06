import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embeddings_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embeddings_table(idx) # B, T, C
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
      logits = logits[:, -1, :] # B, latest Timestep, C
      probs = F.softmax(logits, dim=-1)
      id_next = torch.multinomial(probs, num_samples=1) # B, 1
      idx = torch.cat((idx, id_next), dim=1) # B, T+1
    return idx
  
# test bigram
if __name__ == '__main__':
    model = BigramModel(vocab_size=10)
    idx = torch.randint(0, 10, (3, 5))
    logits, loss = model(idx, targets=idx)
    print(logits.shape)
    print(loss)
    idx = torch.randint(0, 10, (3, 5))
    idx = model.generate(idx, max_tokens=10)
    print(idx.shape)
    print(idx)