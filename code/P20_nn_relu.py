import torchvision.datasets
from torch import nn
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# input = torch.tensor([[1, -0.5],
#                       [-1, 3]])
# output =torch.reshape(input, (-1, 1, 2, 2))
# print(output.shape)

dataset = torchvision.datasets.CIFAR10("../datasets", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

class CZ(nn.Module):
    def __init__(self):
        super(CZ, self).__init__()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output


cz = CZ()

cz = CZ()
writer = SummaryWriter("../log_sigmoid")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = cz(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()

# output = cz(input)
# print(output)


