import sys
import os
import json
import time
import numpy as np
import cv2
import pyttsx3
import dlib
import requests
import bz2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QSlider,
                             QMessageBox, QFrame, QInputDialog, QProgressDialog)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# 禁用OpenCV硬件加速，避免冲突
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)

# 模型文件路径和下载地址配置
MODEL_DIR = os.path.join(os.path.dirname(sys.executable), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
SHAPE_PREDICTOR_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
FACE_RECOGNITION_MODEL_PATH = os.path.join(MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat")

# 国内镜像下载地址（解决网络问题）
MODEL_URLS = {
    "shape_predictor": "https://mirror.ghproxy.com/https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
    "recognition_model": "https://mirror.ghproxy.com/https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2"
}


class ModelDownloader(QThread):
    """模型下载线程（带进度反馈）"""
    progress_updated = pyqtSignal(int)
    _success = False
    _error_msg = ""

    def __init__(self, url, save_path):
        super().__init__()
        self.url = url
        self.save_path = save_path
        self.temp_path = save_path + ".bz2"

    def run(self):
        try:
            # 忽略证书验证（解决网络问题）
            response = requests.get(self.url, stream=True, timeout=30, verify=False)
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(self.temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = int((downloaded_size / total_size) * 100)
                            self.progress_updated.emit(progress)

            # 解压模型文件
            with bz2.BZ2File(self.temp_path, 'rb') as fr, open(self.save_path, 'wb') as fw:
                fw.write(fr.read())

            # 清理临时文件
            if os.path.exists(self.temp_path):
                os.remove(self.temp_path)
            self._success = True

        except Exception as e:
            self._success = False
            self._error_msg = str(e)
        finally:
            self.quit()

    def get_result(self):
        return self._success, self._error_msg


class FaceRecognizer:
    """人脸识别核心类"""
    def __init__(self):
        if not self.check_and_download_models():
            raise Exception("模型准备失败，程序无法启动")

        # 加载dlib模型
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        self.face_rec_model = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

        # 人脸特征库
        self.face_features = {}
        self.features_path = "face_features.json"
        self.load_face_features()

        # 语音冷却设置
        self.cooldown = 5
        self.last_speak_time = {}

    def check_and_download_models(self):
        """检查并下载缺失的模型"""
        missing = []
        if not os.path.exists(SHAPE_PREDICTOR_PATH) or os.path.getsize(SHAPE_PREDICTOR_PATH) < 50 * 1024 * 1024:
            missing.append(("shape_predictor", SHAPE_PREDICTOR_PATH))
        if not os.path.exists(FACE_RECOGNITION_MODEL_PATH) or os.path.getsize(FACE_RECOGNITION_MODEL_PATH) < 10 * 1024 * 1024:
            missing.append(("recognition_model", FACE_RECOGNITION_MODEL_PATH))

        if not missing:
            return True

        # 下载缺失模型
        for model_name, save_path in missing:
            url = MODEL_URLS[model_name]
            progress_dialog = QProgressDialog(f"下载{model_name}模型...", "取消", 0, 100)
            progress_dialog.setWindowTitle("模型下载")
            progress_dialog.setWindowModality(Qt.WindowModal)

            downloader = ModelDownloader(url, save_path)
            downloader.progress_updated.connect(progress_dialog.setValue)
            downloader.start()

            # 等待下载完成
            while downloader.isRunning():
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    downloader.terminate()
                    QMessageBox.warning(None, "取消", "模型下载取消，程序无法运行")
                    return False

            # 检查结果
            success, error = downloader.get_result()
            if not success:
                QMessageBox.critical(None, "失败", f"下载错误：{error}\n请手动下载模型")
                return False

        return True

    def load_face_features(self):
        """加载人脸特征库"""
        if os.path.exists(self.features_path):
            try:
                with open(self.features_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.face_features = {name: np.array(features) for name, features in data.items()}
            except json.JSONDecodeError:
                self.face_features = {}

    def save_face_features(self):
        """保存人脸特征库"""
        data = {name: features.tolist() for name, features in self.face_features.items()}
        with open(self.features_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def detect_and_extract(self, frame):
        """检测人脸并提取特征"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb_frame, 1)
        if not faces:
            return [], []

        bboxes, features = [], []
        for face in faces:
            shape = self.predictor(rgb_frame, face)
            descriptor = self.face_rec_model.compute_face_descriptor(rgb_frame, shape)
            features.append(np.array(descriptor))
            bboxes.append([face.left(), face.top(), face.right(), face.bottom()])
        return bboxes, features

    def recognize(self, features):
        """识别人脸"""
        results = []
        for feat in features:
            if not self.face_features:
                results.append("未知人员")
                continue

            min_dist, recognized_name = float('inf'), "未知人员"
            for name, saved_feat in self.face_features.items():
                dist = np.linalg.norm(feat - saved_feat)
                if dist < min_dist and dist < 0.6:
                    min_dist, recognized_name = dist, name
            results.append(recognized_name)
        return results

    def can_speak(self, name):
        """控制语音播报频率"""
        current_time = time.time()
        if current_time - self.last_speak_time.get(name, 0) > self.cooldown:
            self.last_speak_time[name] = current_time
            return True
        return False


class VoiceEngine:
    """语音播报引擎"""
    def __init__(self):
        self.engine = pyttsx3.init()
        # 配置中文语音
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'chinese' in voice.id.lower():
                self.engine.setProperty('voice', voice.id)
                break
        self.engine.setProperty('rate', 150)  # 语速
        self.engine.setProperty('volume', 0.8)  # 音量

    def set_rate(self, rate):
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume):
        self.engine.setProperty('volume', volume)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()


class FaceRecognitionApp(QMainWindow):
    """主窗口类"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸识别语音播报系统")
        self.setGeometry(100, 100, 900, 700)

        try:
            self.face_recognizer = FaceRecognizer()
        except Exception as e:
            QMessageBox.critical(None, "初始化失败", str(e))
            sys.exit(1)

        self.voice_engine = VoiceEngine()
        self.camera = None
        self.is_running = False

        self.init_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout()

        # 摄像头显示区
        self.camera_label = QLabel("摄像头画面")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid #333;")
        main_layout.addWidget(self.camera_label)

        # 控制按钮区
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始识别")
        self.start_btn.clicked.connect(self.start_recognition)
        self.stop_btn = QPushButton("停止识别")
        self.stop_btn.clicked.connect(self.stop_recognition)
        self.stop_btn.setEnabled(False)
        self.add_btn = QPushButton("添加人脸")
        self.add_btn.clicked.connect(self.add_face)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.add_btn)
        main_layout.addLayout(btn_layout)

        # 语音设置区
        settings_frame = QFrame()
        settings_layout = QVBoxLayout(settings_frame)
        # 语速设置
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("语速:"))
        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setRange(100, 200)
        self.rate_slider.setValue(150)
        self.rate_slider.valueChanged.connect(lambda v: self.voice_engine.set_rate(v))
        rate_layout.addWidget(self.rate_slider)
        settings_layout.addLayout(rate_layout)
        # 音量设置
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("音量:"))
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.valueChanged.connect(lambda v: self.voice_engine.set_volume(v / 100))
        vol_layout.addWidget(self.vol_slider)
        settings_layout.addLayout(vol_layout)
        main_layout.addWidget(settings_frame)

        # 设置中央部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def start_recognition(self):
        """启动识别"""
        if self.camera and self.camera.isOpened():
            self.camera.release()

        # 尝试打开摄像头
        for i in [0, 1, 2]:
            self.camera = cv2.VideoCapture(i)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                break

        if not self.camera or not self.camera.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            return

        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(50)
        QMessageBox.information(self, "提示", "人脸识别已启动")

    def stop_recognition(self):
        """停止识别"""
        self.is_running = False
        self.timer.stop()
        if self.camera:
            self.camera.release()
        self.camera_label.setText("摄像头已关闭")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def add_face(self):
        """添加人脸到特征库"""
        if not self.is_running:
            QMessageBox.warning(self, "提示", "请先启动摄像头")
            return

        ret, frame = self.camera.read()
        if not ret:
            QMessageBox.warning(self, "错误", "无法获取画面")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_recognizer.detector(rgb_frame, 1)
        if len(faces) != 1:
            QMessageBox.warning(self, "提示", "请确保画面中只有一个人脸")
            return

        # 提取特征
        shape = self.face_recognizer.predictor(rgb_frame, faces[0])
        descriptor = self.face_recognizer.face_rec_model.compute_face_descriptor(rgb_frame, shape)
        features = np.array(descriptor)

        # 输入姓名
        name, ok = QInputDialog.getText(self, "添加人脸", "请输入姓名:")
        if ok and name.strip():
            self.face_recognizer.face_features[name.strip()] = features
            self.face_recognizer.save_face_features()
            QMessageBox.information(self, "成功", f"已添加 {name} 到人脸库")

    def update_frame(self):
        """刷新画面并识别"""
        ret, frame = self.camera.read()
        if not ret:
            self.camera_label.setText("无法获取画面")
            return

        frame = cv2.flip(frame, 1)
        bboxes, features = self.face_recognizer.detect_and_extract(frame)
        names = self.face_recognizer.recognize(features)

        # 绘制人脸框和名称
        for bbox, name in zip(bboxes, names):
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if name != "未知人员" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # 语音播报
            if self.face_recognizer.can_speak(name):
                self.voice_engine.speak(f"检测到{name}")

        # 显示到UI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_frame.shape
        q_img = QImage(rgb_frame.data, w, h, w * c, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        self.stop_recognition()
        event.accept()


if __name__ == "__main__":
    os.environ["QT_FONT_DPI"] = "96"
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())