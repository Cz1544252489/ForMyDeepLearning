import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

# train_data = torchvision.datasets.ImageNet("L:\\DL_data", split="train", download=False,
#                                            transform=torchvision.transforms.ToTensor())

# dataset = torchvision.datasets.CIFAR10("../datasets", train=False,
#                                        transform=torchvision.transforms.ToTensor(),
#                                        download=True)
#
# dataloader = DataLoader(dataset, batch_size=64)

vgg16 = torchvision.models.vgg16()

# vgg16.classifier.add_module('7', nn.Linear(1000, 10))
vgg16.classifier[6] = nn.Linear(4096, 10)

print(vgg16)

