import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from tqdm import tqdm


# --------------------------
# 1. 配置参数
# --------------------------
class Config:
    # 数据路径（根据你的文件夹结构修正）
    data_dir = r"D:\机器学习\CNN军事装备识别\war_tech_by_GonTech"  # 正确的根目录
    # 模型参数（补充缺少的input_size）
    input_size = 128  # 关键：添加此参数
    num_classes = 8   # 8个类别（排除imgs文件夹）
    batch_size = 16
    epochs = 30
    lr = 0.001
    weight_decay = 1e-4
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实验对比参数
    experiment_params = {
        "with_bn_dropout": True,
        "without_bn_dropout": False
    }


config = Config()


# --------------------------
# 2. 数据预处理与加载
# --------------------------
def get_data_loaders(data_dir, input_size, batch_size, val_split=0.15, test_split=0.15):
    """
    数据预处理并划分训练集、验证集、测试集
    """
    # 训练集预处理（含数据增强）
    train_transform = Compose([
        Resize((input_size, input_size)),
        RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        RandomRotation(degrees=15),  # 随机旋转
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])

    # 验证集和测试集预处理（无数据增强）
    val_test_transform = Compose([
        Resize((input_size, input_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载完整数据集
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    total_size = len(full_dataset)

    # 划分比例：训练集70%，验证集15%，测试集15%
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size

    # 随机划分
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 重新设置验证集和测试集的transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"数据集划分完成：")
    print(f"训练集样本数：{train_size}, 验证集：{val_size}, 测试集：{test_size}")
    print(f"类别数：{len(full_dataset.classes)}, 类别名称：{full_dataset.classes}")

    return train_loader, val_loader, test_loader


# --------------------------
# 3. CNN模型设计（支持BN和Dropout开关）
# --------------------------
class MilitaryCNN(nn.Module):
    def __init__(self, num_classes, input_size=224, use_bn_dropout=True):
        super(MilitaryCNN, self).__init__()
        self.use_bn_dropout = use_bn_dropout
        self.input_channels = 3

        # 卷积层1
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if use_bn_dropout else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128) if use_bn_dropout else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256) if use_bn_dropout else nn.Identity()
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512) if use_bn_dropout else nn.Identity()
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层输入维度
        self.fc_input_dim = 512 * (input_size // 16) * (input_size // 16)

        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, 1024)
        self.dropout1 = nn.Dropout(0.5) if use_bn_dropout else nn.Identity()
        self.relu5 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5) if use_bn_dropout else nn.Identity()
        self.relu6 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 卷积块1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 卷积块2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 卷积块3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # 卷积块4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu5(x)

        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu6(x)

        x = self.fc3(x)

        return x


# --------------------------
# 4. 训练与评估函数
# --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    """训练模型并记录训练过程"""
    # 记录训练信息
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print("-" * 50)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算平均损失和准确率
        train_avg_loss = train_loss / total_train
        train_acc = train_correct / total_train * 100

        val_avg_loss = val_loss / total_val
        val_acc = val_correct / total_val * 100

        # 记录历史数据
        history["train_loss"].append(train_avg_loss)
        history["val_loss"].append(val_avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # 学习率调整
        if scheduler is not None:
            scheduler.step(val_avg_loss)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()

        print(f"Train Loss: {train_avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")

    # 加载最佳模型权重
    model.load_state_dict(best_model_weights)
    return model, history


def evaluate_model(model, test_loader, device):
    """在测试集上评估模型"""
    model.eval()
    test_correct = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = test_correct / total_test * 100
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    return test_acc


# --------------------------
# 5. 结果可视化函数
# --------------------------
def plot_training_history(history, title_suffix=""):
    """绘制训练损失和准确率曲线"""
    epochs = len(history["train_loss"])
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss")
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss")
    plt.title(f"Loss Curve {title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["train_acc"], label="Train Acc")
    plt.plot(range(1, epochs + 1), history["val_acc"], label="Val Acc")
    plt.title(f"Top-1 Accuracy Curve {title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --------------------------
# 6. 对比实验函数
# --------------------------
def run_comparison_experiments():
    """运行对比实验：有无BN和Dropout"""
    # 加载数据
    train_loader, val_loader, test_loader = get_data_loaders(
        config.data_dir, config.input_size, config.batch_size
    )

    # 存储实验结果
    experiment_results = {}

    for exp_name, use_bn_dropout in config.experiment_params.items():
        print(f"\n{'=' * 60}")
        print(f"实验：{exp_name} (use_bn_dropout={use_bn_dropout})")
        print(f"{'=' * 60}")

        # 创建模型
        model = MilitaryCNN(
            num_classes=config.num_classes,
            input_size=config.input_size,
            use_bn_dropout=use_bn_dropout
        ).to(config.device)

        # 打印模型参数量
        print("\n模型参数量估算：")
        summary(model, (3, config.input_size, config.input_size))

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # 学习率调整策略：移除verbose参数以兼容低版本PyTorch
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # 训练模型
        trained_model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            scheduler, config.epochs, config.device
        )

        # 测试集评估
        test_acc = evaluate_model(trained_model, test_loader, config.device)

        # 保存结果
        experiment_results[exp_name] = {
            "model": trained_model,
            "history": history,
            "test_acc": test_acc
        }

        # 绘制训练曲线
        plot_training_history(history, title_suffix=f"({exp_name})")

    # 打印实验对比总结
    print(f"\n{'=' * 60}")
    print("实验对比总结")
    print(f"{'=' * 60}")
    for exp_name, result in experiment_results.items():
        print(f"{exp_name}: Test Accuracy = {result['test_acc']:.2f}%")

    # 不同参数对比实验（以batch size为例）
    print(f"\n{'=' * 60}")
    print("参数对比实验：不同Batch Size")
    print(f"{'=' * 60}")

    # 测试不同batch size
    batch_sizes = [16, 32, 64]
    batch_size_results = {}

    for bs in batch_sizes:
        print(f"\nBatch Size = {bs}")
        print("-" * 30)

        # 重新加载数据
        train_loader_bs, val_loader_bs, test_loader_bs = get_data_loaders(
            config.data_dir, config.input_size, bs
        )

        # 创建模型（带BN和Dropout）
        model_bs = MilitaryCNN(
            num_classes=config.num_classes,
            input_size=config.input_size,
            use_bn_dropout=True
        ).to(config.device)

        # 训练配置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model_bs.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        # 学习率调整策略：移除verbose参数
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # 训练模型
        trained_model_bs, history_bs = train_model(
            model_bs, train_loader_bs, val_loader_bs, criterion, optimizer,
            scheduler, config.epochs, config.device
        )

        # 测试评估
        test_acc_bs = evaluate_model(trained_model_bs, test_loader_bs, config.device)
        batch_size_results[bs] = test_acc_bs

        # 绘制曲线
        plot_training_history(history_bs, title_suffix=f"(Batch Size={bs})")

    # 打印Batch Size对比结果
    print(f"\nBatch Size对比结果：")
    for bs, acc in batch_size_results.items():
        print(f"Batch Size {bs}: Test Accuracy = {acc:.2f}%")


# --------------------------
# 7. 主函数
# --------------------------
if __name__ == "__main__":
    # 运行对比实验
    run_comparison_experiments()