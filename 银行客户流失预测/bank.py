import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# --------------------------
# 1. 数据加载与预处理（核心修正）
# --------------------------
def load_and_preprocess_data(file_path):
    """加载数据并进行预处理"""
    # 加载数据
    data = pd.read_csv(file_path)

    # 查看数据基本信息
    print("数据集基本信息：")
    print(f"样本数量: {data.shape[0]}, 特征数量: {data.shape[1] - 1}")

    # 处理目标变量（Attrition_Flag）：你的数据中已为1（流失）和0（未流失）
    print("\nAttrition_Flag原始取值分布：")
    print(data['Attrition_Flag'].value_counts(dropna=False))  # 确认原始取值

    # 检查并处理目标变量中的缺失值（如有）
    nan_count = data['Attrition_Flag'].isna().sum()
    if nan_count > 0:
        print(f"\n发现{nan_count}个目标变量缺失值，已自动删除")
        data = data.dropna(subset=['Attrition_Flag'])

    # 确保目标变量为数值型（1/0）
    data['Attrition_Flag'] = data['Attrition_Flag'].astype(int)
    print(f"清洗后流失客户比例: {data['Attrition_Flag'].mean():.4f}")

    # 分离特征和目标变量
    X = data.drop('Attrition_Flag', axis=1)
    y = data['Attrition_Flag']

    # 识别数值特征和分类特征
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    print(f"\n数值特征: {list(numeric_features)}")
    print(f"分类特征: {list(categorical_features)}")

    # 数据预处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # 处理缺失值
        ('scaler', StandardScaler())  # 标准化
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # 处理缺失值
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 独热编码
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 预处理数据
    X_processed = preprocessor.fit_transform(X)

    # 划分训练集和验证集（8:2比例，保持类别分布）
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # 转换为PyTorch张量（处理稀疏矩阵情况）
    if hasattr(X_train, 'toarray'):
        X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
        X_val = torch.tensor(X_val.toarray(), dtype=torch.float32)
    else:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)

    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    print(f"\n预处理后 - 训练集: {X_train.shape}, 验证集: {X_val.shape}")
    return X_train, X_val, y_train, y_val, preprocessor


# --------------------------
# 2. 评价指标计算（保持不变）
# --------------------------
def calculate_metrics(y_pred, y_true):
    """计算准确率、精确率和召回率"""
    # 转换为二值预测（0或1）
    y_pred_binary = (y_pred >= 0.5).float()

    # 准确率
    accuracy = (y_pred_binary == y_true).float().mean().item()

    # 精确率
    true_positive = (y_pred_binary == 1) & (y_true == 1)
    false_positive = (y_pred_binary == 1) & (y_true == 0)
    precision = true_positive.sum().float() / (true_positive.sum() + false_positive.sum()).float() if (
                                                                                                              true_positive.sum() + false_positive.sum()) > 0 else 0.0
    precision = precision.item()

    # 召回率
    false_negative = (y_pred_binary == 0) & (y_true == 1)
    recall = true_positive.sum().float() / (true_positive.sum() + false_negative.sum()).float() if (
                                                                                                           true_positive.sum() + false_negative.sum()) > 0 else 0.0
    recall = recall.item()

    return accuracy, precision, recall


# --------------------------
# 3. 逻辑回归模型（保持不变）
# --------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 线性层

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # sigmoid激活函数映射到[0,1]


def train_logistic_regression(X_train, y_train, X_val, y_val, epochs=200, lr=0.01):
    """训练逻辑回归模型"""
    input_size = X_train.shape[1]
    model = LogisticRegressionModel(input_size)
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 记录训练过程
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }

    for epoch in range(epochs):
        # 训练模式
        model.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 记录训练损失
        history['train_loss'].append(loss.item())

        # 计算训练集指标
        train_acc, train_precision, train_recall = calculate_metrics(outputs, y_train)
        history['train_acc'].append(train_acc)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)

        # 验证模式
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val)
            val_loss = criterion(outputs_val, y_val)
            history['val_loss'].append(val_loss.item())

            val_acc, val_precision, val_recall = calculate_metrics(outputs_val, y_val)
            history['val_acc'].append(val_acc)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)

        # 打印训练进度
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - "
                  f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f} - "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return model, history


# --------------------------
# 4. 线性回归模型（保持不变）
# --------------------------
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 线性层

    def forward(self, x):
        return self.linear(x)  # 无激活函数，输出连续值


def train_linear_regression(X_train, y_train, X_val, y_val, epochs=200, lr=0.01):
    """训练线性回归模型（用于分类任务）"""
    input_size = X_train.shape[1]
    model = LinearRegressionModel(input_size)
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 记录训练过程
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }

    for epoch in range(epochs):
        # 训练模式
        model.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 记录训练损失
        history['train_loss'].append(loss.item())

        # 计算训练集指标（使用0.5作为阈值）
        train_acc, train_precision, train_recall = calculate_metrics(outputs, y_train)
        history['train_acc'].append(train_acc)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)

        # 验证模式
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val)
            val_loss = criterion(outputs_val, y_val)
            history['val_loss'].append(val_loss.item())

            val_acc, val_precision, val_recall = calculate_metrics(outputs_val, y_val)
            history['val_acc'].append(val_acc)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)

        # 打印训练进度
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - "
                  f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f} - "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return model, history


# --------------------------
# 5. 结果可视化（保持不变）
# --------------------------
def plot_training_curves(logistic_history, linear_history):
    """绘制训练过程中的损失和指标曲线"""
    epochs = len(logistic_history['train_loss'])

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), logistic_history['train_loss'], label='Logistic Train')
    plt.plot(range(1, epochs + 1), logistic_history['val_loss'], label='Logistic Val')
    plt.title('Logistic Regression Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), linear_history['train_loss'], label='Linear Train')
    plt.plot(range(1, epochs + 1), linear_history['val_loss'], label='Linear Val')
    plt.title('Linear Regression Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 绘制准确率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), logistic_history['train_acc'], label='Logistic Train')
    plt.plot(range(1, epochs + 1), logistic_history['val_acc'], label='Logistic Val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 绘制精确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), logistic_history['train_precision'], label='Logistic Train')
    plt.plot(range(1, epochs + 1), logistic_history['val_precision'], label='Logistic Val')
    plt.plot(range(1, epochs + 1), linear_history['train_precision'], label='Linear Train')
    plt.plot(range(1, epochs + 1), linear_history['val_precision'], label='Linear Val')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 绘制召回率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), logistic_history['train_recall'], label='Logistic Train')
    plt.plot(range(1, epochs + 1), logistic_history['val_recall'], label='Logistic Val')
    plt.plot(range(1, epochs + 1), linear_history['train_recall'], label='Linear Train')
    plt.plot(range(1, epochs + 1), linear_history['val_recall'], label='Linear Val')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)

    # 绘制两种模型的验证集准确率对比
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), logistic_history['val_acc'], label='Logistic Val')
    plt.plot(range(1, epochs + 1), linear_history['val_acc'], label='Linear Val')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------------------------
# 6. 模型评估与分析（保持不变）
# --------------------------
def evaluate_models(logistic_history, linear_history):
    """评估模型性能并判断拟合情况"""
    print("\n" + "=" * 50)
    print("模型最终性能评估")
    print("=" * 50)

    # 逻辑回归结果
    print("\n逻辑回归:")
    print(f"训练集 - 准确率: {logistic_history['train_acc'][-1]:.4f}, "
          f"精确率: {logistic_history['train_precision'][-1]:.4f}, "
          f"召回率: {logistic_history['train_recall'][-1]:.4f}")
    print(f"验证集 - 准确率: {logistic_history['val_acc'][-1]:.4f}, "
          f"精确率: {logistic_history['val_precision'][-1]:.4f}, "
          f"召回率: {logistic_history['val_recall'][-1]:.4f}")

    # 线性回归结果
    print("\n线性回归:")
    print(f"训练集 - 准确率: {linear_history['train_acc'][-1]:.4f}, "
          f"精确率: {linear_history['train_precision'][-1]:.4f}, "
          f"召回率: {linear_history['train_recall'][-1]:.4f}")
    print(f"验证集 - 准确率: {linear_history['val_acc'][-1]:.4f}, "
          f"精确率: {linear_history['val_precision'][-1]:.4f}, "
          f"召回率: {linear_history['val_recall'][-1]:.4f}")

    # 判断拟合情况
    def check_fitting(train_acc, val_acc):
        diff = abs(train_acc - val_acc)
        if train_acc < 0.7 and val_acc < 0.7:
            return "可能存在欠拟合（训练集和验证集性能都较低）"
        elif diff > 0.1:
            return f"可能存在过拟合（训练集与验证集差异较大: {diff:.4f}）"
        else:
            return f"拟合良好（训练集与验证集性能接近: 差异 {diff:.4f}）"

    print("\n拟合情况分析:")
    print(f"逻辑回归: {check_fitting(logistic_history['train_acc'][-1], logistic_history['val_acc'][-1])}")
    print(f"线性回归: {check_fitting(linear_history['train_acc'][-1], linear_history['val_acc'][-1])}")


# --------------------------
# 7. 主函数（保持不变）
# --------------------------
def main():
    # 设置随机种子，保证结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载并预处理数据
    print("=" * 50)
    print("加载并预处理数据")
    print("=" * 50)
    X_train, X_val, y_train, y_val, preprocessor = load_and_preprocess_data(
        r"D:\机器学习\银行客户流失预测\supply_chain_train.csv"
    )

    # 训练逻辑回归模型
    print("\n" + "=" * 50)
    print("训练逻辑回归模型")
    print("=" * 50)
    logistic_model, logistic_history = train_logistic_regression(
        X_train, y_train, X_val, y_val, epochs=200, lr=0.01
    )

    # 训练线性回归模型
    print("\n" + "=" * 50)
    print("训练线性回归模型")
    print("=" * 50)
    linear_model, linear_history = train_linear_regression(
        X_train, y_train, X_val, y_val, epochs=200, lr=0.01
    )

    # 可视化训练结果
    print("\n" + "=" * 50)
    print("绘制训练曲线")
    print("=" * 50)
    plot_training_curves(logistic_history, linear_history)

    # 评估模型
    evaluate_models(logistic_history, linear_history)

    # 模型对比分析
    print("\n" + "=" * 50)
    print("模型对比分析")
    print("=" * 50)
    print("1. 逻辑回归专为二分类设计，通过sigmoid函数将输出限制在[0,1]区间，具有概率解释性")
    print("2. 线性回归输出为连续值，用于分类时需人为设定阈值（本实验使用0.5）")
    print("3. 逻辑回归使用二元交叉熵损失，更适合分类任务；线性回归使用MSE损失，对异常值更敏感")
    print("4. 通常逻辑回归在分类任务上表现更优，因为其损失函数和激活函数都是为分类问题设计的")


if __name__ == "__main__":
    main()