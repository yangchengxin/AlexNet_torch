import torch
import torch.nn as nn
from AlexNet import AlexNet
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# 解决中文显示问题
# 运行配置参数中的字体（font）为黑体（SimHei）
plt.rcParams['font.sans-serif'] = ['simHei']
# 运行配置参数总的轴（axes）正常显示正负号（minus）
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_TRAIN = 'dataset'
ROOT_TEST = 'dataset'

normalize = transforms.Normalize(
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5]
)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #以0.5的概率来竖直翻转给定的PIL图像
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize,
])

#加载训练数据集
train_datasets = datasets.ImageFolder(ROOT_TRAIN, transform=train_transform)
train_dataloader = DataLoader(train_datasets, batch_size=32, shuffle=True)

val_datasets = datasets.ImageFolder(ROOT_TEST, transform=val_transform)
val_dataloader = DataLoader(val_datasets, batch_size=32, shuffle=True)

#实例化模型对象
model = AlexNet().to(device)

#定义交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()

#定义优化器
optimizer_ = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#学习率每十轮降低为之前的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer_, step_size=10, gamma=0.5)

#定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0.0
    #batch为索引， x和y分别是图片和类别
    for batch, (x, y) in enumerate(dataloader):
        #前向传播
        image, y = x.to(device), y.to(device)
        output = model(image)
        curr_loss = loss_fn(output, y)
        _, pred = torch.max(output, dim=1)
        #计算每个批次的准确率
        curr_acc = torch.sum(y == pred)/output.shape[0]

        #反向传播
        #清空之前的梯度
        optimizer.zero_grad()
        #计算当前的梯度
        curr_loss.backward()
        #根据梯度更新网络参数
        optimizer.step()

        #损失叠加
        loss += curr_loss.item()
        #精度叠加
        current += curr_acc.item()
        n = n + 1

    #训练的平均损失和平均精度
    train_loss = loss / n
    train_acc = current / n
    print('train loss = ' + str(train_loss))
    print('train accuracy = ' + str(train_acc))
    return train_loss, train_acc

#定义验证函数
def val(dataloader, model, loss_fn):
    loss, current, n = 0.0, 0.0, 0.0
    #eval()：如果模型中存在BN和dropout则不启用，以防改变权值
    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            #前向传播
            image, y = x.to(device), y.to(device)
            output = model(image)
            curr_loss = loss_fn(output, y)
            _, pred = torch.max(output, dim=1)
            curr_acc = torch.sum(y == pred) / output.shape[0]
            loss += curr_loss.item()
            current += curr_acc.item()
            n = n + 1

    val_loss = loss / n
    val_acc = current / n
    print('val loss = ' + str(val_loss))
    print('val accuracy = ' + str(val_acc))
    return val_loss, val_acc

#定义画图函数
def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='val loss')

    plt.legend(loc='best')
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.title("训练集和验证集的loss值对比图")
    plt.show()

def plot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train acc')
    plt.plot(val_acc, label='val acc')

    plt.legend(loc='best')
    plt.xlabel('acc')
    plt.ylabel('epoch')
    plt.title("训练集和验证集的acc值对比图")
    plt.show()

#开始训练
loss_train = []
acc_train = []
loss_val = []
acc_val = []

#训练次数
epoch = 200
#用于判断什么时候保存模型
min_acc = 0
for t in range(epoch):
    # lr_scheduler.step()
    print(f"epoch{t+1}-------------------------------")
    #训练模型
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer_)
    #验证模型
    val_loss, val_acc = val(val_dataloader, model, loss_fn)
    print("\n")
    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)
    folder = 'save_model'

    # 保存最好的模型权重
    if val_acc > min_acc:
        if not os.path.exists(folder):
            os.mkdir(folder)
        min_acc = val_acc
        torch.save(model.state_dict(), f"{folder}/model_best.pth")

    if t == epoch - 1:
        torch.save(model.state_dict(), f"{folder}/model_last.pth")
        print("=============训练完毕==============\n" + f"best pth saved as {folder}/model_best.pth\n" + f"last pth saved as {folder}/model_last.pth\n")

plot_loss(loss_train, loss_val)
plot_acc(acc_train, acc_val)



