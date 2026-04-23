# dl-2026-02

## 题目名称

PyTorch 图像分类基础训练：完成模型训练、评估与误差分析

## 这是一道什么题

这道题主要考查：

- PyTorch 张量与 Dataset / DataLoader
- 模型定义、训练循环、验证循环
- loss、optimizer、device 的基础使用
- checkpoint 保存与加载
- 结果可视化与错误分析

本题不要求写出 SOTA 模型，但要求训练流程完整、代码能跑、结果能解释。

## 数据集

任选其一：

- MNIST
- Fashion-MNIST
- CIFAR-10

## 你的任务

### 1. 数据加载与可视化，20 分

要求：

- 使用 `torchvision.datasets` 下载或读取数据
- 正确划分训练集与测试集
- 使用 `DataLoader`
- 保存一张样本可视化图片 `outputs/samples.png`
- 在 README 中说明输入数据的 shape 与类别数量

评分：

- 数据集能正确加载，6 分
- DataLoader 使用正确，5 分
- 可视化结果清楚，4 分
- README 能说明数据 shape、类别、batch size，5 分

### 2. 模型定义与训练循环，25 分

要求：

- 至少实现一个简单神经网络，可以是 MLP 或 CNN
- 支持 CPU / GPU 自动选择
- 完成至少 3 个 epoch 的训练
- 每个 epoch 输出 train loss 与 train accuracy

评分：

- 模型结构能正常前向传播，6 分
- loss 与 optimizer 使用正确，5 分
- 训练循环完整，7 分
- device 处理合理，4 分
- 输出日志清楚，3 分

### 3. 验证、保存与加载 checkpoint，20 分

要求：

- 每个 epoch 后在测试集或验证集上评估
- 保存最佳模型到 `checkpoints/best.pt`
- 编写 `evaluate.py` 加载 checkpoint 并输出最终准确率

评分：

- 验证逻辑正确，6 分
- 最佳 checkpoint 保存逻辑正确，5 分
- `evaluate.py` 可独立运行，5 分
- README 中说明如何复现实验结果，4 分

### 4. 压轴：训练结果诊断与改进实验，25 分

请设计至少 2 组对比实验，并写入 `experiment_report.md`。

可选方向：

- 不同学习率
- 是否使用数据增强
- batch size 大小变化
- optimizer 从 SGD 改为 Adam
- 是否加入 dropout / weight decay

报告必须包含：

- 实验配置表
- 每组实验的准确率
- 至少一张训练曲线图
- 至少 5 张错误分类样本及你的解释
- 你认为模型错在哪里，以及下一步如何改

评分：

- 至少 2 组对比实验真实可运行，6 分
- 配置表与结果表清楚，4 分
- 训练曲线正确生成，4 分
- 错误样本分析具体，不是空话，5 分
- 能提出合理改进方向，6 分

### 5. 代码组织与提交，10 分

建议结构：

```text
.
├── train.py
├── evaluate.py
├── models.py
├── outputs/
├── checkpoints/
├── experiment_report.md
└── README.md
```

评分：

- 文件结构清楚，4 分
- 代码可读性基本良好，3 分
- README 运行说明完整，3 分
