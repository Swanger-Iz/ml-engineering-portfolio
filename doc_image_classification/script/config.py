import os
import torch.nn as nn

epochs = 10
learning_rate = 1e-5     # 1e-5 - 5e-5
batch_size = 64
criterion = nn.CrossEntropyLoss()

classes = ['invoice', 'letter', 'email', 'news_article']
number_of_files = 2000

train_path = os.path.join('..', 'data', 'train')
test_path = os.path.join('..', 'data', 'test')