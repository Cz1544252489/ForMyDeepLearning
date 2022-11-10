import torchvision
from torch.utils.data import DataLoader
from my_model import *


# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 准备数据集
train_dataset = torchvision.datasets.CIFAR10("../datasets",
                                             train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10("../datasets", train=False,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True)

# length 长度
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# 创建网络模型
cz = CZ()
cz = cz.to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(cz.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10


for i in range(epoch):
    print("-----第{}轮训练开始------".format(i+1))

    # 训练步骤开始
    cz.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = cz(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))

    # 测试步骤开始
    cz.eval()
    total_test_loss = 0
    total_accuracy = 0
    # 取消使用梯度
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = cz(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    total_test_step += 1

torch.save(cz, "../logs_train/cz.pth")

