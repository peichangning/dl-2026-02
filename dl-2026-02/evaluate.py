import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import SimpleCNN
import os

def main():
    # 自动选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 检查checkpoint是否存在
    checkpoint_path = 'checkpoints/best.pt'
    if not os.path.exists(checkpoint_path):
        print(f'错误: 未找到模型检查点文件 {checkpoint_path}')
        print('请先运行 train.py 完成训练，生成模型文件。')
        return

    # 加载checkpoint
    print(f'加载模型检查点: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    best_acc = checkpoint['best_acc']

    # 初始化模型并加载权重
    model = SimpleCNN(num_classes=10, use_dropout=args.use_dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'模型加载完成，训练时最佳准确率: {best_acc:.4f}')

    # 数据预处理，与训练时测试集保持一致
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载测试数据集
    print('加载测试数据集...')
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 评估模型
    print('开始评估...')
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_acc = correct / total
    print('=' * 50)
    print(f'最终测试准确率: {final_acc:.4f} ({correct}/{total})')
    print(f'检查点中记录的最佳准确率: {best_acc:.4f}')
    print('=' * 50)

if __name__ == '__main__':
    main()
