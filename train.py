from models import LanguageModel
import torch
from hyperparams import *

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [char_to_idx[ch] for ch in x]
decode = lambda x: ''.join([idx_to_char[i] for i in x])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = LanguageModel(vocab_size=vocab_size)
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optim = torch.optim.Adam(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f'iter {iter:5d} | train loss {losses["train"]:.2f} | val loss {losses["val"]:.2f}')
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
