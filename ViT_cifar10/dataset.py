import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import config

from torchvision.transforms import v2

## Transform version 2
transform_v2 = v2.Compose([
    v2.RandomCrop(size=32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
    v2.ToImage(),    # замена ToTensor()
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),    # [.5, .5, .5]
])

train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform_v2)
test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform_v2)

train_dl = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

train_dataset.classes

