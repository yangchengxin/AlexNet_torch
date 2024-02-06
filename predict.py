import torch
from AlexNet import AlexNet
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

ROOT_TRAIN = 'D:/pycharm/AlexNet/data/train'
ROOT_TEST = 'dataset'

# 将图像的像素值归一化到[-1,1]之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# 加载训练数据集
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

# 如果有NVIDA显卡，转到GPU训练，否则用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型实例化，将模型转到device
model = AlexNet().to(device)

# 加载train.py里训练好的模型
model.load_state_dict(torch.load(r'save_model/model_best.pth'))

# 结果类型
classes = [
    "cat",
    "dog"
]

# 把Tensor转化为图片，方便可视化
show = ToPILImage()

# 进入验证阶段
model.eval()
for i in range(10):
    x, y = val_dataset[i][0], val_dataset[i][1]
    # show()：显示图片
    show(x).show()
    # torch.unsqueeze(input, dim)，input(Tensor)：输入张量，dim (int)：插入维度的索引，最终扩展张量维度为4维
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(x)
        # argmax(input)：返回指定维度最大值的序号
        # 得到预测类别中最高的那一类，再把最高的这一类对应classes中的那一类
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        # 输出预测值与真实值
        print(f'predicted:"{predicted}", actual:"{actual}"')
