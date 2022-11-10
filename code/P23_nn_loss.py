import torch
import torchvision
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch import nn
from torch.utils.data import DataLoader

# inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
# targets = torch.tensor([1, 2, 5], dtype=torch.float32)
#
# inputs = torch.reshape(inputs, (1, 1, 1, 3))
# targets = torch.reshape(targets, (1, 1, 1, 3))
#
# loss = L1Loss()
# result = loss(inputs, targets)
# print(result)
#
# loss_mse = nn.MSELoss()
# result1 = loss_mse(inputs, targets)
# print(result1)

# x = torch.tensor([0.1, 0.2, 0.3])
# y = torch.tensor([1])
# x = torch.reshape(x, (1, 3))
# loss_cross = nn.CrossEntropyLoss()
# result_cross = loss_cross(x, y)
# print(result_cross)


dataset = torchvision.datasets.CIFAR10("../datasets", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

class CZ(nn.Module):
    def __init__(self):
        super(CZ, self).__init__()

        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
cz = CZ()
for data in dataloader:
    imgs, targets = data
    outputs = cz(imgs)
    result_loss = loss(outputs, targets)
    # result_loss.backward()
    print("ok")

