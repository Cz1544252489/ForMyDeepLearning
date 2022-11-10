import torch.optim.optimizer
import torchvision
import time
from torch.utils.data import DataLoader
from my_model import *
from torch.utils.tensorboard import SummaryWriter

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
# 使用gpu
if torch.cuda.is_available():
    cz = cz.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
# 使用gpu
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(cz.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 5

# 添加tensorboard
writer = SummaryWriter("../logs_train")
start_time = time.time()
for i in range(epoch):
    print("-----第{}轮训练开始------".format(i+1))

    # 训练步骤开始
    cz.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = cz(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    cz.eval()
    total_test_loss = 0
    total_accuracy = 0
    # 取消使用梯度
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = cz(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # torch.save(cz.state_dict(), "../logs_train/P27_cz_{}.path".format(i))
    # print("模型已保存")

writer.close()

