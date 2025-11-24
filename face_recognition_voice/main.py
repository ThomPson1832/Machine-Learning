# main.py - 主程序（增强版语音播报人脸识别系统）
import sys
import os
import cv2
import json
import time
import logging
from datetime import datetime
from typing import List, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QInputDialog, QComboBox, QSlider, QGroupBox, QFormLayout,
    QTextEdit, QSplitter, QTabWidget, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

from face_recognizer import FaceRecognizer
from voice_speaker import VoiceSpeaker

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class RecognitionThread(QThread):
    """人脸识别线程"""
    recognition_complete = pyqtSignal(list)

    def __init__(self, recognizer, frame):
        super().__init__()
        self.recognizer = recognizer
        self.frame = frame.copy()

    def run(self):
        try:
            results = self.recognizer.recognize(self.frame)
            self.recognition_complete.emit(results)
        except Exception as e:
            logging.error(f"识别线程错误: {e}")
            self.recognition_complete.emit([])


class FaceRecognitionApp(QMainWindow):
    """主应用程序"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎭 智能语音播报人脸识别系统")
        self.setGeometry(100, 100, 1400, 900)

        # 初始化组件
        self.recognizer = FaceRecognizer()
        self.speaker = VoiceSpeaker()

        # 状态变量
        self.is_recognizing = False
        self.is_camera_available = False
        self.last_speak_time = {}
        self.recognition_results = []

        # 加载配置
        self.config = self.load_config()
        self.apply_config()

        # 初始化UI
        self.init_ui()
        self.init_camera()

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

        logging.info("应用程序初始化完成")

    def load_config(self):
        """加载配置"""
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载配置失败: {e}")
            return self.get_default_config()

    def get_default_config(self):
        """获取默认配置"""
        return {
            "mode": "完整模式",
            "volume": 0.8,
            "rate": 150,
            "enable_unknown_alert": True,
            "speak_cooldown": 5,
            "camera_index": 0,
            "face_database": "face_database.json",
            "recognition_threshold": 0.6
        }

    def save_config(self):
        """保存配置"""
        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存配置失败: {e}")

    def apply_config(self):
        """应用配置"""
        # 语音设置
        self.speaker.set_volume(self.config.get('volume', 0.8))
        self.speaker.set_rate(self.config.get('rate', 150))

        # 应用设置
        self.speak_cooldown = self.config.get('speak_cooldown', 5)
        self.enable_unknown_alert = self.config.get('enable_unknown_alert', True)
        self.current_mode = self.config.get('mode', '完整模式')

    def init_ui(self):
        """初始化用户界面"""
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: white;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QComboBox, QSlider {
                background-color: #3c3c3c;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QGroupBox {
                color: #4CAF50;
                font-weight: bold;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                border: 1px solid #555;
                border-radius: 4px;
                font-family: 'Courier New';
            }
        """)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 左侧视频区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 视频显示
        self.video_label = QLabel("摄像头未启动")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("QLabel { background: black; color: white; font-size: 16px; }")

        # 控制按钮
        control_layout = QHBoxLayout()
        self.recognize_btn = QPushButton("🎧 开始识别")
        self.capture_btn = QPushButton("📸 采集人脸")
        self.clear_db_btn = QPushButton("🗑️ 清空数据库")

        control_layout.addWidget(self.recognize_btn)
        control_layout.addWidget(self.capture_btn)
        control_layout.addWidget(self.clear_db_btn)

        # 状态显示
        self.status_label = QLabel("🔴 系统就绪")
        self.status_label.setStyleSheet("QLabel { font-size: 16px; color: #ff6b6b; }")

        left_layout.addWidget(self.video_label)
        left_layout.addLayout(control_layout)
        left_layout.addWidget(self.status_label)

        # 右侧控制面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setAlignment(Qt.AlignTop)

        # 语音设置组
        voice_group = QGroupBox("🎵 语音设置")
        voice_layout = QFormLayout(voice_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["简洁模式", "完整模式", "安全模式", "静音模式"])
        self.mode_combo.setCurrentText(self.current_mode)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(self.config.get('volume', 0.8) * 100))

        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setRange(50, 300)
        self.rate_slider.setValue(self.config.get('rate', 150))

        voice_layout.addRow("播报模式:", self.mode_combo)
        voice_layout.addRow("音量调节:", self.volume_slider)
        voice_layout.addRow("语速调节:", self.rate_slider)

        # 识别设置组
        recog_group = QGroupBox("🔍 识别设置")
        recog_layout = QFormLayout(recog_group)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 10)
        self.threshold_slider.setValue(int(self.config.get('recognition_threshold', 0.6) * 10))

        self.cooldown_slider = QSlider(Qt.Horizontal)
        self.cooldown_slider.setRange(1, 30)
        self.cooldown_slider.setValue(self.config.get('speak_cooldown', 5))

        recog_layout.addRow("识别阈值:", self.threshold_slider)
        recog_layout.addRow("播报间隔:", self.cooldown_slider)

        # 日志显示
        log_group = QGroupBox("📝 系统日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        # 数据库信息
        db_group = QGroupBox("💾 数据库信息")
        db_layout = QVBoxLayout(db_group)
        self.db_info_label = QLabel("加载中...")
        db_layout.addWidget(self.db_info_label)

        right_layout.addWidget(voice_group)
        right_layout.addWidget(recog_group)
        right_layout.addWidget(db_group)
        right_layout.addWidget(log_group)

        # 分割布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([800, 400])

        layout.addWidget(splitter)

        # 连接信号
        self.connect_signals()

        # 更新数据库信息
        self.update_database_info()

    def connect_signals(self):
        """连接信号槽"""
        self.recognize_btn.clicked.connect(self.toggle_recognition)
        self.capture_btn.clicked.connect(self.capture_face)
        self.clear_db_btn.clicked.connect(self.clear_database)

        self.mode_combo.currentTextChanged.connect(self.on_mode_change)
        self.volume_slider.valueChanged.connect(self.on_volume_change)
        self.rate_slider.valueChanged.connect(self.on_rate_change)
        self.threshold_slider.valueChanged.connect(self.on_threshold_change)
        self.cooldown_slider.valueChanged.connect(self.on_cooldown_change)

    def init_camera(self):
        """初始化摄像头"""
        try:
            camera_index = self.config.get('camera_index', 0)
            self.cap = cv2.VideoCapture(camera_index)

            if self.cap.isOpened():
                self.is_camera_available = True
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.log_message("✅ 摄像头初始化成功")
            else:
                self.is_camera_available = False
                self.log_message("❌ 摄像头初始化失败")

        except Exception as e:
            self.is_camera_available = False
            self.log_message(f"❌ 摄像头初始化错误: {e}")

    def toggle_recognition(self):
        """切换识别状态"""
        if not self.is_camera_available:
            QMessageBox.warning(self, "警告", "摄像头不可用，请检查连接")
            return

        if not self.is_recognizing:
            # 开始识别
            self.is_recognizing = True
            self.recognize_btn.setText("🛑 停止识别")
            self.recognize_btn.setStyleSheet("background-color: #ff6b6b;")
            self.status_label.setText("🟢 识别中...")
            self.status_label.setStyleSheet("QLabel { color: #4CAF50; }")
            self.timer.start(100)  # 10 FPS
            self.log_message("🎯 开始人脸识别")
        else:
            # 停止识别
            self.is_recognizing = False
            self.recognize_btn.setText("🎧 开始识别")
            self.recognize_btn.setStyleSheet("background-color: #4CAF50;")
            self.status_label.setText("🔴 识别已停止")
            self.status_label.setStyleSheet("QLabel { color: #ff6b6b; }")
            self.timer.stop()
            self.log_message("⏹️ 停止人脸识别")

    def process_frame(self):
        """处理视频帧"""
        if not self.is_camera_available:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.log_message("❌ 无法读取摄像头画面")
            return

        # 水平翻转
        frame = cv2.flip(frame, 1)

        if self.is_recognizing:
            # 在子线程中进行识别
            self.recognition_thread = RecognitionThread(self.recognizer, frame)
            self.recognition_thread.recognition_complete.connect(self.on_recognition_complete)
            self.recognition_thread.start()
        else:
            # 只显示画面
            self.display_frame(frame)

    def on_recognition_complete(self, results):
        """识别完成回调"""
        self.recognition_results = results

        # 获取当前帧并显示结果
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.display_frame(frame, results)
            self.handle_voice_announce(results)

    def display_frame(self, frame, results=None):
        """显示视频帧和识别结果"""
        display_frame = frame.copy()

        if results:
            known_count = 0
            unknown_count = 0

            for name, bbox, confidence in results:
                x, y, w, h = bbox

                # 设置颜色
                if name == "未知人员":
                    color = (0, 0, 255)  # 红色
                    unknown_count += 1
                else:
                    color = (0, 255, 0)  # 绿色
                    known_count += 1

                # 绘制矩形
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                # 绘制标签
                label = f"{name} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_frame, (x, y - label_size[1] - 10),
                              (x + label_size[0], y), color, -1)
                cv2.putText(display_frame, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 显示统计信息
            stats_text = f"已知: {known_count} | 未知: {unknown_count}"
            cv2.putText(display_frame, stats_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 转换为Qt格式显示
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def handle_voice_announce(self, results):
        """处理语音播报"""
        if not results or self.current_mode == "静音模式":
            return

        current_time = time.time()
        names_to_speak = []

        for name, _, confidence in results:
            # 检查冷却时间
            last_time = self.last_speak_time.get(name, 0)
            if current_time - last_time > self.speak_cooldown:
                names_to_speak.append(name)
                self.last_speak_time[name] = current_time

        if names_to_speak:
            # 过滤未知人员（根据配置）
            known_names = [name for name in names_to_speak if name != "未知人员"]
            unknown_names = [name for name in names_to_speak if name == "未知人员"]

            # 构建播报列表
            speak_names = known_names
            if self.enable_unknown_alert and unknown_names:
                speak_names.extend(unknown_names)

            if speak_names:
                self.speaker.speak_face_result(speak_names, self.current_mode, self.enable_unknown_alert)
                self.log_message(f"🎵 播报: {', '.join(speak_names)}")

    def capture_face(self):
        """采集人脸"""
        if not self.is_camera_available:
            QMessageBox.warning(self, "警告", "摄像头不可用")
            return

        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "错误", "无法获取摄像头画面")
            return

        frame = cv2.flip(frame, 1)

        # 检测是否有人脸
        faces = self.recognizer.detect_faces(frame)
        if len(faces) == 0:
            QMessageBox.warning(self, "提示", "未检测到人脸，请调整位置")
            return
        elif len(faces) > 1:
            QMessageBox.warning(self, "提示", "检测到多个人脸，请确保只有一个人脸在画面中")
            return

        # 输入姓名
        name, ok = QInputDialog.getText(self, "采集人脸", "请输入姓名:")
        if ok and name.strip():
            success, message = self.recognizer.capture_face(frame, name.strip())
            if success:
                QMessageBox.information(self, "成功", message)
                self.log_message(f"✅ {message}")
                self.update_database_info()
            else:
                QMessageBox.warning(self, "失败", message)
                self.log_message(f"❌ {message}")

    def clear_database(self):
        """清空数据库"""
        reply = QMessageBox.question(self, "确认", "确定要清空所有人脸数据吗？此操作不可恢复！",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.recognizer.face_database = {}
            self.recognizer.save_face_database()
            self.update_database_info()
            self.log_message("🗑️ 人脸数据库已清空")
            QMessageBox.information(self, "完成", "数据库已清空")

    def update_database_info(self):
        """更新数据库信息"""
        db_info = self.recognizer.get_database_info()
        info_text = f"""
        总人数: {db_info['total_faces']}
        注册名单: {', '.join(db_info['names']) if db_info['names'] else '无'}
        最后更新: {db_info['last_update']}
        """
        self.db_info_label.setText(info_text)

    def log_message(self, message):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)

        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        logging.info(message)

    # 配置变更处理
    def on_mode_change(self, mode):
        self.current_mode = mode
        self.config['mode'] = mode
        self.save_config()
        self.log_message(f"🔧 切换模式: {mode}")

    def on_volume_change(self, value):
        volume = value / 100.0
        self.speaker.set_volume(volume)
        self.config['volume'] = volume
        self.save_config()

    def on_rate_change(self, value):
        self.speaker.set_rate(value)
        self.config['rate'] = value
        self.save_config()

    def on_threshold_change(self, value):
        threshold = value / 10.0
        self.recognizer.recognition_threshold = threshold
        self.config['recognition_threshold'] = threshold
        self.save_config()
        self.log_message(f"🔧 识别阈值调整为: {threshold}")

    def on_cooldown_change(self, value):
        self.speak_cooldown = value
        self.config['speak_cooldown'] = value
        self.save_config()
        self.log_message(f"🔧 播报间隔调整为: {value}秒")

    def closeEvent(self, event):
        """关闭事件"""
        self.timer.stop()
        if hasattr(self, 'cap'):
            self.cap.release()
        self.speaker.stop()
        self.save_config()
        self.log_message("🔚 应用程序已退出")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用程序属性
    app.setApplicationName("智能语音播报人脸识别系统")
    app.setApplicationVersion("2.0.0")

    window = FaceRecognitionApp()
    window.show()

    sys.exit(app.exec_())