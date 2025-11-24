import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# 导入Keras相关模块（基于TensorFlow后端）
from tensorflow.keras import models, layers, losses, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint


# 1. 数据预处理（适配Keras输入格式：channel_last，归一化到0-1）
def preprocess_data():
    # 加载MNIST数据集（Keras内置接口，直接返回训练/测试集）
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 预处理：1. 增加通道维度（Keras卷积层需 (28,28,1) 格式）；2. 归一化到0-1；3. 转换数据类型
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255.0

    # 打印数据集信息（与原代码保持一致）
    print(f"训练数据集长度：{len(x_train)}")
    print(f"测试数据集长度：{len(x_test)}")

    return (x_train, y_train), (x_test, y_test)


# 2. 定义MNIST识别模型（复刻原PyTorch的卷积+全连接结构，用Keras Sequential搭建）
def build_mnist_model():
    model = models.Sequential([
        # 卷积层1：10个5x5卷积核，激活函数ReLU（对应原PyTorch的conv1+relu）
        layers.Conv2D(10, (5, 5), activation="relu", input_shape=(28, 28, 1)),
        # 池化层1：2x2最大池化（对应原PyTorch的maxpool1）
        layers.MaxPooling2D((2, 2)),
        # 卷积层2：20个5x5卷积核，激活函数ReLU（对应原PyTorch的conv2+relu）
        layers.Conv2D(20, (5, 5), activation="relu"),
        # 池化层2：2x2最大池化（对应原PyTorch的maxpool2）
        layers.MaxPooling2D((2, 2)),
        # 展平层：将卷积输出转为一维（对应原PyTorch的x.view()）
        layers.Flatten(),
        # 全连接层1：128个神经元（对应原PyTorch的linear1）
        layers.Dense(128, activation="relu"),
        # 全连接层2：64个神经元（对应原PyTorch的linear2）
        layers.Dense(64, activation="relu"),
        # 输出层：10个神经元（对应0-9数字，无激活（Keras损失函数会处理））
        layers.Dense(10)
    ])

    # 编译模型：指定优化器、损失函数（与原PyTorch一致）
    # SparseCategoricalCrossentropy：标签为整数（无需独热编码，对应原PyTorch的CrossEntropyLoss）
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.14),  # 学习率保持0.14
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),  # from_logits=True：输出未经过softmax
        metrics=["accuracy"]  # 训练时同时计算准确率
    )

    return model


# 3. 模型训练（Keras用model.fit()自动处理训练循环，无需手动写反向传播）
def train_model(model, x_train, y_train, x_test, y_test):
    # 创建model目录（避免保存路径错误）
    if not os.path.exists('./model'):
        os.makedirs('./model')

    # 模型保存回调：每100个批次（batch）保存一次模型（对应原代码的每100轮保存）
    checkpoint = ModelCheckpoint(
        filepath="./model/keras_mnist_model.h5",  # Keras模型保存格式（.h5）
        monitor="loss",  # 监测损失值
        save_best_only=False,  # 不只保存最优模型，每100批次都保存
        save_weights_only=False,  # 保存完整模型（含结构+权重）
        verbose=1,  # 保存时打印提示
        period=100  # 每100个批次保存一次
    )

    # 开始训练（对应原代码的5轮训练）
    model.fit(
        x_train, y_train,
        batch_size=64,  # 批次大小与原代码一致
        epochs=5,  # 训练轮次与原代码一致
        validation_data=(x_test, y_test),  # 用测试集做验证（对应原代码的test()函数）
        callbacks=[checkpoint],  # 加入保存模型的回调
        shuffle=True  # 训练前打乱数据（对应原代码的shuffle=True）
    )


# 4. 自定义手写图片识别（流程与原代码一致，适配Keras输入格式）
def test_mydata(model):
    # 创建test目录（避免路径错误）
    if not os.path.exists('./test'):
        os.makedirs('./test')

    # 读取并预处理自定义图片
    try:
        # 注意：原代码文件名是test_there.png，错误提示是test_two.png，这里统一为test_two.png（可自行修改）
        image = Image.open('./test/test_two.png')
    except FileNotFoundError:
        print("错误：未找到test_two.png，请将图片放入./test目录")
        return

    # 预处理：1.  resize到28x28；2. 转灰度图；3. 归一化到0-1；4. 增加维度（适配Keras输入：(1,28,28,1)）
    image = image.resize((28, 28)).convert('L')  # 转灰度
    image_np = np.array(image).astype("float32") / 255.0  # 归一化
    image_input = image_np.reshape((1, 28, 28, 1))  # 增加batch和通道维度

    # 模型预测（Keras用model.predict()）
    logits = model.predict(image_input, verbose=0)  # 输出未经过softmax的logits
    probability = np.exp(logits) / np.sum(np.exp(logits))  # 转为概率（softmax）
    predict_label = np.argmax(probability)  # 取概率最大的类别
    max_prob = np.max(probability)  # 最大概率值

    # 打印结果+可视化（与原代码一致）
    print(f"自定义图片预测结果：{predict_label}，最大概率：{max_prob:.2f}")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.title(f"预测结果：{predict_label}")
    plt.imshow(image, cmap='gray')  # 灰度图显示
    plt.show()


# 5. 主函数（保持原代码的功能选择逻辑）
if __name__ == '__main__':
    # 加载并预处理数据
    (x_train, y_train), (x_test, y_test) = preprocess_data()

    # 构建模型
    model = build_mnist_model()

    # 加载已保存的模型（如果存在）
    if os.path.exists('./model/keras_mnist_model.h5'):
        model = models.load_model('./model/keras_mnist_model.h5')
        print("已加载预训练模型")

    # 功能选择（与原代码一致）
    choice = input("请选择功能（1-训练+MNIST测试；2-识别自定义图片）：")
    if choice == '1':
        train_model(model, x_train, y_train, x_test, y_test)
    elif choice == '2':
        test_mydata(model)
    else:
        print("输入错误，请选择1或2")