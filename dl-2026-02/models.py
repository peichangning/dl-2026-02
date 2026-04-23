import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    简单的CNN模型，用于CIFAR-10图像分类
    """
    def __init__(self, num_classes=10, use_dropout=False):
        super(SimpleCNN, self).__init__()
        self.use_dropout = use_dropout
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        # 经过两次池化后，特征图尺寸为 8x8，通道数64
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout层（可选）
        if self.use_dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一层卷积+池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积+池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 64 * 8 * 8)
        
        # Dropout（如果启用）
        if self.use_dropout:
            x = self.dropout(x)
            
        # 全连接层
        x = F.relu(self.fc1(x))
        
        if self.use_dropout:
            x = self.dropout(x)
            
        x = self.fc2(x)
        return x
