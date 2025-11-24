import os
import sys
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('Qt5Agg')  # 确保与PyQt5兼容
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Keras与PyQt5模块
from tensorflow.keras import models, layers, losses, optimizers, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor

# 中文字体设置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
matplotlib.rcParams['axes.unicode_minus'] = False


# -------------------------- 数据与模型基础功能 --------------------------
def preprocess_data():
    """加载并预处理MNIST数据集"""
    try:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # 调整形状并归一化
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255.0
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255.0
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        print(f"数据预处理错误: {e}")
        return None, None


def build_mnist_model():
    """构建MNIST识别模型（兼容低版本TensorFlow）"""
    model = models.Sequential([
        Input(shape=(28, 28, 1)),
        layers.Conv2D(10, (5, 5), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(20, (5, 5), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10)
    ])
    # 显式编译模型
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.14),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model


# -------------------------- 训练与预测线程（避免UI卡顿） --------------------------
class TrainingCallback(Callback):
    """训练过程回调，更新UI日志"""

    def __init__(self, update_signal):
        super().__init__()
        self.update_signal = update_signal

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {epoch + 1} - 损失: {logs.get('loss', 0):.4f}, "
        msg += f"准确率: {logs.get('accuracy', 0):.4f}, "
        msg += f"验证损失: {logs.get('val_loss', 0):.4f}, "
        msg += f"验证准确率: {logs.get('val_accuracy', 0):.4f}\n"
        self.update_signal.emit(msg)

    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            logs = logs or {}
            self.update_signal.emit(f"批次 {batch} - 损失: {logs.get('loss', 0):.4f}\n")


class TrainThread(QThread):
    """模型训练线程"""
    update_signal = pyqtSignal(str)
    finish_signal = pyqtSignal()

    def __init__(self, model, x_train, y_train, x_test, y_test):
        super().__init__()
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def run(self):
        try:
            if not os.path.exists('./model'):
                os.makedirs('./model')

            # 模型保存检查点（兼容低版本TensorFlow）
            checkpoint = ModelCheckpoint(
                filepath="./model/keras_mnist_model.h5",
                monitor="loss",
                save_best_only=False,
                save_weights_only=False,
                verbose=0,
                period=100  # 低版本用period参数
            )

            # 开始训练
            self.model.fit(
                self.x_train, self.y_train,
                batch_size=64,
                epochs=5,
                validation_data=(self.x_test, self.y_test),
                callbacks=[checkpoint, TrainingCallback(self.update_signal)],
                shuffle=True,
                verbose=0
            )
        except Exception as e:
            self.update_signal.emit(f"训练错误: {str(e)}\n")
        finally:
            self.finish_signal.emit()


class PredictThread(QThread):
    """预测线程"""
    result_signal = pyqtSignal(int, float, np.ndarray)

    def __init__(self, model, image_np):
        super().__init__()
        self.model = model
        self.image_np = image_np

    def run(self):
        try:
            # 预处理图像
            image_input = self.image_np.reshape((1, 28, 28, 1)).astype("float32") / 255.0
            # 预测
            logits = self.model.predict(image_input, verbose=0)
            probability = np.exp(logits) / np.sum(np.exp(logits))
            predict_label = np.argmax(probability)
            max_prob = np.max(probability)
            self.result_signal.emit(predict_label, max_prob, self.image_np)
        except Exception as e:
            print(f"预测错误: {e}")
            self.result_signal.emit(-1, 0.0, None)


# -------------------------- 手写数字画布（核心交互组件） --------------------------
class HandwritingCanvas(QLabel):
    """手写数字绘制画布"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)  # 10倍放大28x28，便于书写
        self.setStyleSheet("border: 2px solid #333;")  # 边框
        self.clear_canvas()
        self.is_drawing = False
        self.last_point = QPoint()

    def clear_canvas(self):
        """清空画布"""
        self.canvas = QImage(self.size(), QImage.Format_RGB32)
        self.canvas.fill(Qt.white)  # 白色背景
        self.update()

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        """鼠标移动事件（绘制）"""
        if event.buttons() & Qt.LeftButton and self.is_drawing:
            painter = QPainter(self.canvas)
            # 画笔设置（黑色、粗细5px、圆角）
            painter.setPen(QPen(Qt.black, 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self.is_drawing = False

    def paintEvent(self, event):
        """绘制画布内容"""
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.canvas)

    def get_handwriting_np(self):
        """将手写内容转为28x28的numpy数组（兼容MNIST格式）"""
        try:
            # 1. 缩放到28x28
            scaled_img = self.canvas.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # 2. 转为灰度图
            gray_img = scaled_img.convertToFormat(QImage.Format_Grayscale8)
            # 3. 转为numpy数组（安全处理内存）
            width = gray_img.width()
            height = gray_img.height()
            ptr = gray_img.bits()
            ptr.setsize(height * width)  # 显式设置内存大小，避免访问越界
            img_np = np.frombuffer(ptr, np.uint8).reshape((height, width))
            # 4. 反色（MNIST是黑底白字）
            img_np = 255 - img_np  # 白色背景→黑色背景，黑色笔迹→白色笔迹
            return img_np
        except Exception as e:
            print(f"图像转换错误: {e}")
            return np.zeros((28, 28), dtype=np.uint8)


# -------------------------- 主窗口 --------------------------
class MNISTApp(QMainWindow):
    """主应用窗口"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.init_ui()
        self.load_data()
        self.load_model()

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle('MNIST手写数字识别')
        self.setGeometry(100, 100, 1000, 700)  # 窗口大小

        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧：控制与日志区
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, 1)

        # 右侧：手写与结果区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, 1)

        # -------------------------- 左侧面板 --------------------------
        # 标题
        title_label = QLabel('手写数字识别系统')
        title_label.setAlignment(Qt.AlignCenter)
        font = title_label.font()
        font.setPointSize(14)
        font.setBold(True)
        title_label.setFont(font)
        left_layout.addWidget(title_label)

        # 功能按钮
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton('训练模型')
        self.train_btn.clicked.connect(self.start_training)
        self.clear_btn = QPushButton('清空画布')
        self.clear_btn.clicked.connect(self.clear_handwriting)
        self.predict_btn = QPushButton('识别数字')
        self.predict_btn.clicked.connect(self.start_prediction)
        btn_layout.addWidget(self.train_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.predict_btn)
        left_layout.addLayout(btn_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # 日志区域
        left_layout.addWidget(QLabel('运行日志:'))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text)

        # -------------------------- 右侧面板 --------------------------
        # 手写画布
        self.handwriting_canvas = HandwritingCanvas()
        right_layout.addWidget(self.handwriting_canvas)

        # 结果显示
        self.result_label = QLabel('请在画布上手写数字，然后点击"识别数字"')
        self.result_label.setAlignment(Qt.AlignCenter)
        result_font = self.result_label.font()
        result_font.setPointSize(12)
        self.result_label.setFont(result_font)
        right_layout.addWidget(self.result_label)

    def load_data(self):
        """加载MNIST数据集"""
        self.log_text.append('正在加载MNIST数据集...')
        (self.x_train, self.y_train), (self.x_test, self.y_test) = preprocess_data()
        if self.x_train is not None:
            self.log_text.append(f'数据集加载完成 - 训练集: {len(self.x_train)} 张, 测试集: {len(self.x_test)} 张')
        else:
            self.log_text.append('数据集加载失败，请检查网络连接')
            QMessageBox.warning(self, '警告', '数据集加载失败，请检查网络连接')

    def load_model(self):
        """加载或创建模型"""
        self.log_text.append('正在初始化模型...')
        try:
            if os.path.exists('./model/keras_mnist_model.h5'):
                self.model = models.load_model('./model/keras_mnist_model.h5')
                # 强制重新编译（解决版本兼容问题）
                self.model.compile(
                    optimizer=optimizers.SGD(learning_rate=0.14),
                    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=["accuracy"]
                )
                self.log_text.append('已加载预训练模型')
            else:
                self.model = build_mnist_model()
                self.log_text.append('未找到预训练模型，已创建新模型')
        except Exception as e:
            self.log_text.append(f'模型加载失败: {str(e)}，将使用新模型')
            self.model = build_mnist_model()

    def start_training(self):
        """开始训练模型"""
        if self.model is None or self.x_train is None:
            QMessageBox.warning(self, '警告', '模型或数据未初始化完成')
            return

        # 禁用按钮，显示进度条
        self.train_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_text.append('开始模型训练...')

        # 启动训练线程
        self.train_thread = TrainThread(self.model, self.x_train, self.y_train, self.x_test, self.y_test)
        self.train_thread.update_signal.connect(self.update_log)
        self.train_thread.finish_signal.connect(self.training_finished)
        self.train_thread.start()

    def update_log(self, message):
        """更新日志并刷新进度条"""
        self.log_text.append(message)
        self.log_text.moveCursor(self.log_text.textCursor().End)  # 滚动到最新内容
        # 简单更新进度（5个epoch，每个epoch约20%）
        current = self.progress_bar.value()
        self.progress_bar.setValue(min(current + 1, 100))

    def training_finished(self):
        """训练完成回调"""
        self.log_text.append('模型训练完成')
        self.train_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        # 保存最终模型
        try:
            self.model.save('./model/keras_mnist_model.h5')
            self.log_text.append('模型已保存到 ./model 目录')
        except Exception as e:
            self.log_text.append(f'模型保存失败: {str(e)}')

    def clear_handwriting(self):
        """清空手写画布"""
        self.handwriting_canvas.clear_canvas()
        self.result_label.setText('画布已清空，请重新手写数字')

    def start_prediction(self):
        """开始识别手写数字"""
        # 获取手写图像
        img_np = self.handwriting_canvas.get_handwriting_np()
        # 检查是否有有效内容（避免全黑/全白）
        if np.sum(img_np) < 10:  # 阈值判断是否有笔迹
            QMessageBox.warning(self, '警告', '请先在画布上手写数字！')
            return
        if self.model is None:
            QMessageBox.warning(self, '警告', '模型未准备好，请先训练模型')
            return

        self.log_text.append('正在识别手写数字...')
        self.predict_btn.setEnabled(False)

        # 启动预测线程
        self.predict_thread = PredictThread(self.model, img_np)
        self.predict_thread.result_signal.connect(self.show_prediction_result)
        self.predict_thread.start()

    def show_prediction_result(self, label, prob, img_np):
        """显示预测结果"""
        if label == -1:
            self.log_text.append('识别失败，请重试')
            self.result_label.setText('识别失败，请重试')
        else:
            self.log_text.append(f'识别成功 - 预测数字: {label}, 概率: {prob:.2f}')
            self.result_label.setText(f'预测结果: {label} (概率: {prob:.2f})')
        self.predict_btn.setEnabled(True)


# -------------------------- 程序入口 --------------------------
if __name__ == '__main__':
    # 创建必要目录
    for dir_name in ['./model', './test']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # 解决高DPI显示问题
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    window = MNISTApp()
    window.show()
    sys.exit(app.exec_())