import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10("../datasets", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class CZ(nn.Module):
    def __init__(self):
        super(CZ, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output


cz = CZ()
writer = SummaryWriter("../log_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = cz(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()

