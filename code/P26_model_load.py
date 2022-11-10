import torch
import torchvision
from P26_model_save import *

# 方式1
# model = torch.load("vgg_method1.pth")
# print(model)

# 方式2
# vgg16 = torchvision.models.vgg16()
# vgg16.load_state_dict(torch.load("vgg_method2.pth"))
# print(vgg16)

# 需要原Class类型，尝试使用 from P26_model_save import * 解决
model = torch.load("cz.pth")
print(model)

