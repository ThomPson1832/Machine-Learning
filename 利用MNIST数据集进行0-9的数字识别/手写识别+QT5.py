import os
import sys
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('Qt5Agg')  # 改用PyQt5兼容的后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 导入Keras相关模块
from tensorflow.keras import models, layers, losses, optimizers, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

# 导入PyQt5相关模块
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QFileDialog, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 1. 数据预处理
def preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 增加通道维度并归一化
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


# 2. 定义模型（修正Input层警告）
def build_mnist_model():
    model = models.Sequential([
        Input(shape=(28, 28, 1)),  # 用Input层替代input_shape参数
        layers.Conv2D(10, (5, 5), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(20, (5, 5), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10)
    ])

    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.14),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model


# 自定义回调类，用于在训练过程中更新UI
class TrainingCallback(Callback):
    def __init__(self, update_signal):
        super().__init__()
        self.update_signal = update_signal

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        message = f"Epoch {epoch + 1} - 损失: {logs.get('loss'):.4f}, "
        message += f"准确率: {logs.get('accuracy'):.4f}, "
        message += f"验证损失: {logs.get('val_loss'):.4f}, "
        message += f"验证准确率: {logs.get('val_accuracy'):.4f}\n"
        self.update_signal.emit(message)

    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:  # 每100个批次更新一次
            logs = logs or {}
            self.update_signal.emit(f"批次 {batch} - 损失: {logs.get('loss'):.4f}\n")


# 训练线程
class TrainThread(QThread):
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finish_signal = pyqtSignal()

    def __init__(self, model, x_train, y_train, x_test, y_test):
        super().__init__()
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = 5
        self.batch_size = 64

    def run(self):
        # 确保model目录存在
        if not os.path.exists('./model'):
            os.makedirs('./model')

        # 创建检查点
        checkpoint = ModelCheckpoint(
            filepath="./model/keras_mnist_model.h5",
            monitor="loss",
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
            save_freq=100 * self.batch_size  # 替换period（旧参数），每100批次保存
        )

        # 训练模型
        self.model.fit(
            self.x_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.x_test, self.y_test),
            callbacks=[checkpoint, TrainingCallback(self.update_signal)],
            shuffle=True,
            verbose=0
        )

        self.finish_signal.emit()


# 预测线程
class PredictThread(QThread):
    result_signal = pyqtSignal(int, float, np.ndarray)

    def __init__(self, model, image_path):
        super().__init__()
        self.model = model
        self.image_path = image_path

    def run(self):
        try:
            # 读取并预处理图像
            image = Image.open(self.image_path)
            image = image.resize((28, 28)).convert('L')  # 转为28x28灰度图
            image_np = np.array(image).astype("float32") / 255.0
            image_input = image_np.reshape((1, 28, 28, 1))

            # 预测
            logits = self.model.predict(image_input, verbose=0)
            probability = np.exp(logits) / np.sum(np.exp(logits))
            predict_label = np.argmax(probability)
            max_prob = np.max(probability)

            self.result_signal.emit(predict_label, max_prob, np.array(image))
        except Exception as e:
            self.result_signal.emit(-1, 0.0, None)


# 图像显示画布
class ImageCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes.axis('off')  # 关闭坐标轴

    def display_image(self, image_np, title=""):
        self.axes.clear()
        self.axes.imshow(image_np, cmap='gray')
        self.axes.set_title(title)
        self.axes.axis('off')
        self.fig.tight_layout()
        self.draw()


# 主窗口类
class MNISTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.init_ui()
        self.load_data()
        self.load_model()

    def init_ui(self):
        # 设置窗口
        self.setWindowTitle('MNIST手写数字识别')
        self.setGeometry(100, 100, 900, 700)

        # 主部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧面板 - 控制和信息
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, 1)

        # 右侧面板 - 图像显示
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, 1)

        # 左侧面板内容
        # 标题
        title_label = QLabel('MNIST手写数字识别系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_font = title_label.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        left_layout.addWidget(title_label)

        # 按钮区域
        button_layout = QHBoxLayout()

        self.train_btn = QPushButton('训练模型')
        self.train_btn.clicked.connect(self.start_training)
        button_layout.addWidget(self.train_btn)

        self.load_img_btn = QPushButton('加载图片')
        self.load_img_btn.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_img_btn)

        self.predict_btn = QPushButton('识别图片')
        self.predict_btn.clicked.connect(self.start_prediction)
        self.predict_btn.setEnabled(False)
        button_layout.addWidget(self.predict_btn)

        left_layout.addLayout(button_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # 日志区域
        log_label = QLabel('训练日志:')
        left_layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text)

        # 右侧面板内容
        self.image_canvas = ImageCanvas(self, width=5, height=5, dpi=100)
        right_layout.addWidget(self.image_canvas)

        # 预测结果
        self.result_label = QLabel('预测结果将显示在这里')
        self.result_label.setAlignment(Qt.AlignCenter)
        result_font = self.result_label.font()
        result_font.setPointSize(12)
        self.result_label.setFont(result_font)
        right_layout.addWidget(self.result_label)

        # 变量初始化
        self.current_image_path = None

    def load_data(self):
        try:
            self.log_text.append('正在加载MNIST数据集...')
            (self.x_train, self.y_train), (self.x_test, self.y_test) = preprocess_data()
            self.log_text.append(f'数据集加载完成 - 训练集: {len(self.x_train)}, 测试集: {len(self.x_test)}')
        except Exception as e:
            self.log_text.append(f'加载数据集出错: {str(e)}')
            QMessageBox.critical(self, '错误', f'加载数据集出错: {str(e)}')

    def load_model(self):
        self.log_text.append('正在加载模型...')
        try:
            if os.path.exists('./model/keras_mnist_model.h5'):
                self.model = models.load_model('./model/keras_mnist_model.h5')
                self.log_text.append('已加载预训练模型')
            else:
                self.model = build_mnist_model()
                self.log_text.append('未找到预训练模型，已创建新模型')
        except Exception as e:
            self.log_text.append(f'加载模型出错: {str(e)}')
            self.model = build_mnist_model()
            self.log_text.append('已创建新模型')

    def start_training(self):
        if self.model is None:
            QMessageBox.warning(self, '警告', '模型未初始化')
            return

        self.train_btn.setEnabled(False)
        self.load_img_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.log_text.append('开始模型训练...')

        # 创建并启动训练线程
        self.train_thread = TrainThread(self.model, self.x_train, self.y_train, self.x_test, self.y_test)
        self.train_thread.update_signal.connect(self.update_train_log)
        self.train_thread.finish_signal.connect(self.training_finished)
        self.train_thread.start()

    def update_train_log(self, message):
        self.log_text.append(message)
        # 自动滚动到底部
        self.log_text.moveCursor(self.log_text.textCursor().End)

        # 简单更新进度条
        current_value = self.progress_bar.value()
        new_value = min(current_value + 1, 100)
        self.progress_bar.setValue(new_value)

    def training_finished(self):
        self.log_text.append('模型训练完成')
        self.train_btn.setEnabled(True)
        self.load_img_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        # 保存最终模型
        try:
            self.model.save('./model/keras_mnist_model.h5')
            self.log_text.append('模型已保存')
        except Exception as e:
            self.log_text.append(f'保存模型出错: {str(e)}')

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', './test', '图像文件 (*.png *.jpg *.jpeg *.bmp)'
        )

        if file_path:
            self.current_image_path = file_path
            self.predict_btn.setEnabled(True)

            # 显示选中的图片
            try:
                image = Image.open(file_path)
                image = image.resize((28, 28)).convert('L')
                self.image_canvas.display_image(np.array(image), '待识别图片')
                self.result_label.setText('请点击"识别图片"按钮进行识别')
                self.log_text.append(f'已加载图片: {os.path.basename(file_path)}')
            except Exception as e:
                self.log_text.append(f'加载图片出错: {str(e)}')
                QMessageBox.critical(self, '错误', f'加载图片出错: {str(e)}')

    def start_prediction(self):
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            QMessageBox.warning(self, '警告', '请先加载有效的图片')
            return

        if self.model is None:
            QMessageBox.warning(self, '警告', '模型未初始化')
            return

        self.log_text.append('正在进行图片识别...')
        self.predict_btn.setEnabled(False)

        # 创建并启动预测线程
        self.predict_thread = PredictThread(self.model, self.current_image_path)
        self.predict_thread.result_signal.connect(self.show_prediction_result)
        self.predict_thread.start()

    def show_prediction_result(self, label, prob, image_np):
        if label == -1 or image_np is None:
            self.log_text.append('识别失败')
            self.result_label.setText('识别失败，请重试')
            self.predict_btn.setEnabled(True)
            return

        self.log_text.append(f'识别完成 - 预测结果: {label}, 概率: {prob:.2f}')
        self.result_label.setText(f'预测结果: {label} (概率: {prob:.2f})')
        self.image_canvas.display_image(image_np, f'预测结果: {label} (概率: {prob:.2f})')
        self.predict_btn.setEnabled(True)


if __name__ == '__main__':
    # 确保必要的目录存在
    if not os.path.exists('./model'):
        os.makedirs('./model')
    if not os.path.exists('./test'):
        os.makedirs('./test')

    app = QApplication(sys.argv)
    window = MNISTApp()
    window.show()
    sys.exit(app.exec_())