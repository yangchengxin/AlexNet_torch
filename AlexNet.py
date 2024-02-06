import torch
import torchvision
from torch import nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        #特征提取层
        self.CONV = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        #将多维的张量进行平坦化处理
        self.flatten = nn.Flatten()

        #全连接层
        self.FC = nn.Sequential(
            # 全连接层1
            nn.Linear(in_features=6 * 6 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 全连接层2
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 全连接层3
            nn.Linear(in_features=4096, out_features=1000),
            nn.Dropout(0.5),
            # 全连接层4 分类层，几分类就是降维到几
            nn.Linear(in_features=1000, out_features=10),
        )

    def forward(self, x):
        x = self.CONV(x)
        x = self.flatten(x)
        x = self.FC(x)
        return x

if __name__ == "__main__":
    x = torch.randn([1,3,224,224])
    model = AlexNet()
    y = model(x)
    print(x)