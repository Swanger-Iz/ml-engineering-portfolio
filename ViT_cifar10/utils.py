import torch
import random



torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')