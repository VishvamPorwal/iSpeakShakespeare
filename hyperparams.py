import torch

learning_rate = 3e-4
max_iters = 1000
eval_iters = 100
batch_size = 32
block_size = 128
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
n_embd = 256
n_head = 8
n_layer = 8
dropout = 0.2
eval_interval = 100
