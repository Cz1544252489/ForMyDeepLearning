import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../datasets", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

class CZ(nn.Module):
    def __init__(self):
        super(CZ, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


cz = CZ()

writer = SummaryWriter("../set")

step = 0
for data in dataloader:
    imgs, targets = data
    output = cz(imgs)
    print(imgs.shape)
    print(output.shape)
    # imgs = torch.reshape(imgs, (-1, 3, 32, 32))
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 32, 32))
    # torch.Size([64, 3, 32, 32])
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
print(cz)



