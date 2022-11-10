from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("../train/cats/cat.30.jpg")
print(img)

# ToTensor
trans_Totensor = transforms.ToTensor()
img_tensor = trans_Totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
# output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_tensor[0][2][0])
trans_norm = transforms.Normalize([1, 2, 3], [2, 1, 3])
img_norm = trans_norm(img_tensor)
print(img_norm[0][2][0])
writer.add_image("Normalize", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((256, 256))
# img PIL -> resize -> img_resize tensor
img_resize = trans_resize(img)
# img_resize PIL -> Totensor -> img_resize tensor
img_resize = trans_Totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(type(img_resize))

# compose - resize -2
trans_resize_2 = transforms.Resize((50, 50))
trans_compose = transforms.Compose([trans_resize_2, trans_Totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop((50, 50))
trans_compose_2 = transforms.Compose([trans_random, trans_Totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()


