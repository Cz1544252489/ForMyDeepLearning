import torch
from torch import nn


class CZ(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


cz = CZ()
x = torch.tensor(1.0)
output = cz(x)
print(output)


