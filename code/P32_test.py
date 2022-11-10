import torch
import torchvision.transforms
from PIL import Image
from torch import nn

image_path = "C:\\Users\\chenz\\pytorch\\train\\dogs\\dog.3.jpg"

image = Image.open(image_path)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

# case 1
model = torch.load("../logs_train/cz.pth", map_location=torch.device("cpu"))
# case 2
# model = torch.load("../logs_train/cz.pth")
print(model)

image = torch.reshape(image, (1, 3, 32, 32))

model.eval()
with torch.no_grad():
    # case 1
    output = model(image)
    # case 2
    # output = model(image.cuda())

print(output)
print(output.argmax(1))


