import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import SimpleCNN

# 设置matplotlib中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def set_seed(seed=42):
    """设置随机种子，保证实验可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CIFAR-10 图像分类训练')
    parser.add_argument('--lr', default=0.01, type=float, help='学习率')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='训练轮数')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'], help='优化器类型')
    parser.add_argument('--use_dropout', action='store_true', help='是否使用Dropout')
    parser.add_argument('--use_augmentation', action='store_true', help='是否使用数据增强')
    args = parser.parse_args()

    # 创建必要的文件夹
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # 设置随机种子
    set_seed()

    # 自动选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 数据预处理
    # 测试集的预处理固定
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 训练集预处理，根据是否使用数据增强调整
    if args.use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        train_transform = test_transform

    # 加载CIFAR-10数据集
    print('加载数据集...')
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # CIFAR-10类别名称
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 1. 数据样本可视化
    print('生成样本可视化图片...')
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i in range(8):
        # 反归一化，恢复原始图片
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(classes[labels[i]])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/samples.png', dpi=100)
    plt.close()
    print(f'样本可视化已保存到 outputs/samples.png')

    # 初始化模型、损失函数和优化器
    model = SimpleCNN(num_classes=10, use_dropout=args.use_dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=0.9,
            weight_decay=5e-4 if args.use_dropout else 0
        )
    else: # Adam
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr,
            weight_decay=5e-4 if args.use_dropout else 0
        )

    # 训练记录
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_acc = 0.0

    print('开始训练...')
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()

            # 统计
            running_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # 验证阶段
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        error_samples = [] # 收集错误分类样本

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 统计
                running_test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                # 收集前5个错误样本用于后续分析
                for i in range(inputs.size(0)):
                    if predicted[i] != labels[i] and len(error_samples) < 5:
                        error_samples.append((
                            inputs[i].cpu(), 
                            classes[labels[i]], 
                            classes[predicted[i]]
                        ))

        epoch_test_loss = running_test_loss / total_test
        epoch_test_acc = correct_test / total_test
        test_losses.append(epoch_test_loss)
        test_accs.append(epoch_test_acc)

        # 保存最佳模型
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args
            }, 'checkpoints/best.pt')

        # 输出训练日志
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'  训练 Loss: {epoch_train_loss:.4f}, 训练准确率: {epoch_train_acc:.4f}')
        print(f'  测试 Loss: {epoch_test_loss:.4f}, 测试准确率: {epoch_test_acc:.4f}')
        print(f'  最佳测试准确率: {best_acc:.4f}')
        print('-' * 60)

    # 生成训练曲线
    print('生成训练曲线...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss曲线
    ax1.plot(train_losses, label='训练Loss')
    ax1.plot(test_losses, label='测试Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss 变化曲线')
    ax1.legend()

    # Accuracy曲线
    ax2.plot(train_accs, label='训练准确率')
    ax2.plot(test_accs, label='测试准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('准确率变化曲线')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('outputs/train_curve.png', dpi=100)
    plt.close()
    print(f'训练曲线已保存到 outputs/train_curve.png')

    # 生成错误样本可视化
    print('生成错误样本可视化...')
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        img, true_label, pred_label = error_samples[i]
        # 反归一化
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'真实: {true_label}\n预测: {pred_label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/error_samples.png', dpi=100)
    plt.close()
    print(f'错误样本已保存到 outputs/error_samples.png')

    print(f'训练完成! 最佳准确率: {best_acc:.4f}')

if __name__ == '__main__':
    main()
