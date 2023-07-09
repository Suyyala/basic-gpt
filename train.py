
import torch
import torch.nn as nn
from torch.nn import functional as F
from bigram import BigramModel
from attention import AttnHead
from decoder import DecoderModel

# load data
with open('data/input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(chars)
print(vocab_size)

# tokenization
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
print(stoi)
print(itos)

encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: "".join([itos[i] for i in x])
print(encode('hello world'))
print(decode(encode('hello world')))


data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])


# split data
n = int(0.9 * len(data))
print(n)
train_d = data[:n]
val_d = data[n:]
print(train_d.shape, val_d.shape)

# hyper params
batch_size = 128
block_size = 32
num_attn_blocks = 16
n_embd = 128
learning_rate = 1e-3
max_iters = 5000
eval_interval = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# to make results reproducible
torch.manual_seed(42)

def get_batch_data(data_type = 'train'):
  data_set = train_d if data_type == 'train' else  val_d
  ix = torch.randint(low=0, high=len(data_set) - block_size, size=(batch_size, ))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x, y

def estimate_loss(model, data_type='train'):
    model.eval()
    with torch.no_grad():
        x, y = get_batch_data(data_type=data_type)
        logits, loss = model(x, y)
    
    model.train()
    return loss.item()

# instantiate model
# model = BigramModel(vocab_size=vocab_size)
model = DecoderModel(vocab_size=vocab_size, num_attn_blocks= num_attn_blocks, block_size=block_size, embed_dim=n_embd)
m = model.to(device)

# training loop
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# training loop
for steps in range(max_iters):
  xb, yb = get_batch_data()

  # evaluate loss on train and validation set
  if steps % eval_interval == 0:
    train_loss = estimate_loss(m, data_type='train')
    val_loss = estimate_loss(m, data_type='val')
    print(f"step: {steps}, train_loss: {train_loss}, val_loss: {val_loss}")
     
  
  # evaluate model
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(loss.item())

# generate text
print('generating text...')
context = torch.zeros((1, 1), dtype=torch.long)
idx = m.generate(idx=context, max_tokens=500)[0].tolist()
print(idx)
print(decode(idx))

# save model
print('saving model...')
torch.save(m.state_dict(), 'models/bigram_model.pth')

