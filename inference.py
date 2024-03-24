from models import LanguageModel
import torch
from hyperparams import *
from train import decode, vocab_size

model = LanguageModel(vocab_size=vocab_size)
model.load_state_dict(torch.load('model.pth'))
model = model.to(device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
