import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# 1. 数据预处理（仅保留必要转换，注释简化）
transform = transforms.Compose([
    transforms.ToTensor(),  # 灰度图(0-255)转Tensor(0-1)
])

# 2. 加载MNIST数据集（保留核心加载逻辑）
# 训练集
train_data = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# 测试集
test_data = MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

# 打印数据集长度（保留关键信息）
print(f"训练数据集长度：{len(train_data)}")
print(f"测试数据集长度：{len(test_data)}")


# 3. 定义MNIST识别模型（保留核心网络结构，删除冗余注释）
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        # 卷积层+池化层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1)
        self.maxpool2 = nn.MaxPool2d(2)
        # 全连接层
        self.linear1 = nn.Linear(320, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积+池化+激活
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        # 展平（适配全连接层输入）
        x = x.view(x.size(0), -1)
        # 全连接层前向传播
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


# 4. 初始化模型、损失函数、优化器（保留核心组件）
model = MnistModel()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适配多分类）
optimizer = torch.optim.SGD(model.parameters(), lr=0.14)  # SGD优化器


# 5. 模型训练函数（删除冗余变量，保留关键逻辑）
def train():
    for index, (input, target) in enumerate(train_loader):
        # 前向传播：模型预测
        y_predict = model(input)
        # 计算损失
        loss = criterion(y_predict, target)
        # 反向传播：梯度清零→损失回传→参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每100轮打印损失+保存模型（保留关键 checkpoint 逻辑）
        if index % 100 == 0:
            # 先创建model目录（避免之前的路径错误）
            if not os.path.exists('./model'):
                os.makedirs('./model')
            # 保存模型和优化器参数
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")
            print(f"训练轮次：{index}，当前损失：{loss.item()}")


# 6. 模型测试函数（MNIST测试集验证准确率，保留核心计算）
def test():
    correct = 0  # 正确预测数
    total = 0  # 总样本数
    with torch.no_grad():  # 测试阶段不计算梯度（节省资源）
        for input, target in test_loader:
            output = model(input)
            # 获取预测结果（概率最大的类别）
            _, predict = torch.max(output.data, dim=1)
            # 累计总数和正确数
            total += target.size(0)
            correct += (predict == target).sum().item()
    # 打印准确率
    print(f"MNIST测试集准确率：{correct / total:.6f}")


# 7. 自定义手写图片识别函数（保留核心识别+可视化）
def test_mydata():
    # 先创建test目录（避免路径错误）
    if not os.path.exists('./test'):
        os.makedirs('./test')
    # 读取并预处理自定义图片
    try:
        image = Image.open('./test/test_there.png')
    except FileNotFoundError:
        print("错误：未找到test_two.png，请将图片放入./test目录")
        return
    # 转换为28*28灰度图
    image = image.resize((28, 28)).convert('L')
    # 转为Tensor并调整维度（适配模型输入：[batch, channel, h, w]）
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # 增加batch维度（从[1,28,28]→[1,1,28,28]）

    # 模型预测
    with torch.no_grad():
        output = model(image_tensor)
        probability, predict = torch.max(output.data, dim=1)

    # 打印结果+可视化
    print(f"自定义图片预测结果：{predict[0]}，最大概率：{probability[0]:.2f}")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.title(f"预测结果：{int(predict[0])}")
    plt.imshow(image, cmap='gray')  # 灰度图显示
    plt.show()


# 8. 主函数（合并重复的__name__判断，统一入口）
if __name__ == '__main__':
    # 先加载已保存的模型（如果存在）
    if os.path.exists('./model/model.pkl'):
        model.load_state_dict(torch.load("./model/model.pkl"))
        print("已加载预训练模型")

    # 选择功能：1-训练+MNIST测试；2-仅识别自定义图片
    choice = input("请选择功能（1-训练+MNIST测试；2-识别自定义图片）：")
    if choice == '1':
        # 训练5轮（原15轮太长，调整为5轮平衡效果和速度）
        for i in range(5):
            print(f"\n————————第{i + 1}轮训练+测试开始——————")
            train()
            test()
    elif choice == '2':
        test_mydata()
    else:
        print("输入错误，请选择1或2")