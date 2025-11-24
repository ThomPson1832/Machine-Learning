# -*- coding: UTF-8 -*-
# 利用鸢尾花数据集，实现前向传播、反向传播，可视化loss曲线

# 导入所需模块
import tensorflow as tf  # 导入TensorFlow深度学习框架
from sklearn import datasets  # 导入sklearn的数据集模块，用于获取鸢尾花数据
from matplotlib import pyplot as plt  # 导入matplotlib，用于可视化
import numpy as np  # 导入numpy，用于数据处理

# 导入数据，分别为输入特征和标签
# 鸢尾花数据集包含4个特征(花萼长度、花萼宽度、花瓣长度、花瓣宽度)和3个类别
x_data = datasets.load_iris().data  # 加载特征数据，shape为(150, 4)
y_data = datasets.load_iris().target  # 加载标签数据，shape为(150,)，值为0,1,2分别代表三种鸢尾花

# 随机打乱数据（因为原始数据是按类别顺序排列的，顺序不打乱会影响模型训练效果）
# seed: 随机数种子，保证每次运行时打乱的顺序一致，便于结果复现
np.random.seed(116)  # 使用相同的seed，确保输入特征和标签打乱后仍一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)  # 设置TensorFlow的随机种子

# 将打乱后的数据集分割为训练集和测试集
# 训练集为前120条数据，测试集为后30条数据
x_train = x_data[:-30]  # 训练特征集
y_train = y_data[:-30]  # 训练标签集
x_test = x_data[-30:]  # 测试特征集
y_test = y_data[-30:]  # 测试标签集

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
# 将numpy数组转换为TensorFlow的float32类型
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 构建数据集迭代器
# from_tensor_slices函数使输入特征和标签值一一对应
# batch(32)表示每32条数据作为一个训练批次
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 生成神经网络的参数
# 输入层：4个神经元（对应4个特征）
# 输出层：3个神经元（对应3个类别）
# 用tf.Variable()标记参数为可训练参数
# truncated_normal生成截断正态分布的随机数，避免出现过大的值
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))  # 权重矩阵，shape为[4,3]
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))  # 偏置项，shape为[3]

lr = 0.1  # 学习率，控制参数更新的幅度
train_loss_results = []  # 存储每轮的loss值，用于后续绘制loss曲线
test_acc = []  # 存储每轮的准确率，用于后续绘制准确率曲线
epoch = 500  # 训练轮数，整个数据集循环500次
loss_all = 0  # 用于累加每轮中所有批次的loss值

# 训练部分
# 外层循环：遍历整个数据集的次数（epoch）
for epoch in range(epoch):  # 数据集级别的循环，每个epoch循环一次完整数据集
    # 内层循环：遍历每个批次（batch）
    for step, (x_train, y_train) in enumerate(train_db):  # batch级别的循环，每个step处理一个batch
        # with结构用于记录梯度信息，实现自动求导
        with tf.GradientTape() as tape:  # 记录梯度计算过程
            # 前向传播计算：y = x * w + b
            y = tf.matmul(x_train, w1) + b1  # 矩阵乘法运算，得到未激活的输出
            y = tf.nn.softmax(y)  # 通过softmax函数将输出转换为概率分布（使输出值在0-1之间且和为1）
            # 将标签转换为独热编码（one-hot encoding）
            # 因为是3分类，所以depth=3，例如标签0会转换为[1,0,0]
            y_ = tf.one_hot(y_train, depth=3)
            # 计算损失函数（均方误差）
            loss = tf.reduce_mean(tf.square(y_ - y))  # 均方误差：mean(sum((y_true - y_pred)^2))
            loss_all += loss.numpy()  # 累加当前批次的loss值

        # 反向传播：计算损失函数对各参数的梯度
        grads = tape.gradient(loss, [w1, b1])  # 计算loss对w1和b1的偏导数

        # 梯度下降更新参数：w = w - lr * gradient
        w1.assign_sub(lr * grads[0])  # 更新权重w1
        b1.assign_sub(lr * grads[1])  # 更新偏置b1

    # 每轮训练结束后，打印当前轮的平均loss
    print("Epoch {}, loss: {}".format(epoch, loss_all / 4))  # 120条数据，每批32条，共4批
    train_loss_results.append(loss_all / 4)  # 记录当前轮的平均loss
    loss_all = 0  # 重置loss_all，为下一轮计算做准备

    # 测试部分：每轮训练后，在测试集上评估模型性能
    total_correct = 0  # 记录预测正确的样本数
    total_number = 0  # 记录测试的总样本数
    for x_test, y_test in test_db:
        # 使用当前参数进行前向传播计算
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)  # 转换为概率分布
        pred = tf.argmax(y, axis=1)  # 找到概率最大的索引，即预测的类别
        # 将预测结果转换为与标签相同的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 比较预测结果与真实标签，相等则为1（正确），否则为0（错误）
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)  # 计算当前批次中正确的样本数
        total_correct += int(correct)  # 累加所有批次的正确样本数
        total_number += x_test.shape[0]  # 累加总样本数（x_test的行数）

    # 计算准确率
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴标签：训练轮次
plt.ylabel('Loss')  # y轴标签：损失值
plt.plot(train_loss_results, label="$Loss$")  # 绘制loss曲线
plt.legend()  # 显示图例
plt.show()  # 显示图像

# 绘制 Accuracy 曲线
plt.title('Accuracy Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴标签：训练轮次
plt.ylabel('Accuracy')  # y轴标签：准确率
plt.plot(test_acc, label="$Accuracy$")  # 绘制准确率曲线
plt.legend()  # 显示图例
plt.show()  # 显示图像
