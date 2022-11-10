from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import cv2
import os
from torchvision import  transforms

writer = SummaryWriter("logs")
image_path = "../train/cats/cat.10.jpg"
img_PIL = Image.open(image_path)

# 把PIL图片转化为np的array格式
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

# transforms 的使用
# 把PIL图片转化为tensor格式
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img_PIL)

# 使用opencv，把ndarray转化为tensor
cv_img = cv2.imread(image_path)
tensor_img1 = tensor_trans(cv_img)

writer.add_image("test", img_array, 2, dataformats='HWC')
# y = x
for i in range(100):
    writer.add_scalar("y=3x", i, i)

writer.close()