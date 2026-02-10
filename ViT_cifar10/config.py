
import torch.nn as nn


cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# hyperparams
batch_size = 128
epochs = 10
learning_rate = 3e-4
patch_size = 4 # segments 
num_classes = 10
image_size = 32
channels = 3
embed_dim = 256
num_heads = 8    # for multi head attention
depth = 6   # means 6 transformer blocks
mlp_dim = 512    # multilinear perceptron layers
drop_rate = .1


criterion = nn.CrossEntropyLoss()