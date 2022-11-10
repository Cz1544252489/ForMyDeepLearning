import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d

vgg16 = torchvision.models.vgg16()

# 保存方式1，模型结构和模型参数
torch.save(vgg16, "vgg_method1.pth")

# 保存方式2，模型参数（官网推荐）
torch.save(vgg16.state_dict(), "vgg_method2.pth")

# 陷阱
class CZ(nn.Module):
    def __init__(self):
        super(CZ, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output


cz = CZ()
torch.save(cz, "cz.pth")

