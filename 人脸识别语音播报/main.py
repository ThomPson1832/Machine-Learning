import sys
import os
import cv2
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import List, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QInputDialog, QComboBox, QSlider, QGroupBox, QFormLayout,
    QTextEdit, QSplitter, QTabWidget, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

import face_recognition

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class VoiceSpeaker:
    """语音合成器（轻量版）"""

    def __init__(self):
        self.engine = None
        self.volume = 0.8
        self.rate = 150
        self.is_speaking = False
        self.speech_mutex = QMutex()
        self.init_engine()

    def init_engine(self):
        """初始化语音引擎"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            # 简化语音选择，减少初始化时间
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id)

            self.engine.setProperty('volume', self.volume)
            self.engine.setProperty('rate', self.rate)
        except Exception as e:
            logging.error(f"语音引擎初始化失败: {e}")
            self.engine = None

    def set_volume(self, volume):
        """设置音量"""
        self.volume = volume
        if self.engine:
            self.engine.setProperty('volume', volume)

    def set_rate(self, rate):
        """设置语速"""
        self.rate = rate
        if self.engine:
            self.engine.setProperty('rate', rate)

    def speak_face_result(self, names, mode, enable_unknown_alert):
        """根据识别结果进行语音播报"""
        if not self.engine or not names:
            return

        # 简化语音逻辑，减少计算
        known_names = [name for name in names if name != "未知人员"]
        unknown_count = names.count("未知人员")

        if not known_names and not (enable_unknown_alert and unknown_count > 0):
            return

        speak_text = ""
        if known_names:
            speak_text = f"识别到{len(known_names)}位"
        if enable_unknown_alert and unknown_count > 0:
            speak_text += f"未知{unknown_count}位"

        if speak_text and not self.is_speaking:
            self.speak(speak_text)

    def speak(self, text):
        """语音播报"""
        if not self.engine or self.is_speaking:
            return

        def _speak():
            self.is_speaking = True
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logging.error(f"语音播报失败: {e}")
            finally:
                self.is_speaking = False

        import threading
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()

    def stop(self):
        """停止语音播报"""
        if self.engine and self.is_speaking:
            try:
                self.engine.stop()
            except:
                pass
            self.is_speaking = False


class FaceRecognizer:
    """人脸识别器（性能优化版）"""

    def __init__(self):
        self.face_database = {}
        self.face_database_path = "face_database.json"
        self.recognition_threshold = 0.6
        self.db_mutex = QMutex()
        self.last_recognition_time = 0
        self.recognition_interval = 1.0  # 识别间隔1秒
        self.load_face_database()

    def load_face_database(self):
        """加载人脸数据库"""
        try:
            if os.path.exists(self.face_database_path):
                with QMutexLocker(self.db_mutex):
                    with open(self.face_database_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for name, encoding in data.items():
                            self.face_database[name] = np.array(encoding)
                logging.info(f"加载人脸数据库成功，共 {len(self.face_database)} 人")
        except Exception as e:
            logging.error(f"加载人脸数据库失败: {e}")
            self.face_database = {}

    def save_face_database(self):
        """保存人脸数据库"""
        try:
            with QMutexLocker(self.db_mutex):
                data = {name: encoding.tolist() for name, encoding in self.face_database.items()}
                with open(self.face_database_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存人脸数据库失败: {e}")

    def fast_detect_faces(self, frame):
        """快速人脸检测"""
        try:
            # 使用小尺寸图像进行快速检测
            small_frame = cv2.resize(frame, (320, 240))
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # 使用HOG模型（最快）
            face_locations = face_recognition.face_locations(
                rgb_frame,
                number_of_times_to_upsample=0,  # 不进行上采样
                model="hog"
            )

            # 转换坐标格式
            converted_locations = []
            for (top, right, bottom, left) in face_locations:
                # 缩放回原图坐标
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 240
                x = int(left * scale_x)
                y = int(top * scale_y)
                w = int((right - left) * scale_x)
                h = int((bottom - top) * scale_y)
                converted_locations.append((x, y, w, h))

            return converted_locations

        except Exception as e:
            logging.error(f"快速人脸检测失败: {e}")
            return []

    def recognize_fast(self, frame):
        """快速识别人脸"""
        try:
            current_time = time.time()
            if current_time - self.last_recognition_time < self.recognition_interval:
                return []

            self.last_recognition_time = current_time

            # 快速检测人脸
            face_locations = self.fast_detect_faces(frame)
            if not face_locations:
                return []

            # 使用小尺寸图像提取特征
            small_frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # 转换坐标到小尺寸
            small_face_locations = []
            for (x, y, w, h) in face_locations:
                scale_x = 640 / frame.shape[1]
                scale_y = 480 / frame.shape[0]
                small_top = int(y * scale_y)
                small_right = int((x + w) * scale_x)
                small_bottom = int((y + h) * scale_y)
                small_left = int(x * scale_x)
                small_face_locations.append((small_top, small_right, small_bottom, small_left))

            # 提取特征
            face_encodings = face_recognition.face_encodings(rgb_frame, small_face_locations)

            results = []
            with QMutexLocker(self.db_mutex):
                for i, (location, encoding) in enumerate(zip(face_locations, face_encodings)):
                    x, y, w, h = location

                    name = "未知人员"
                    min_distance = float('inf')

                    # 快速比对（限制最大比对次数）
                    max_compares = min(10, len(self.face_database))  # 最多比对10个
                    db_items = list(self.face_database.items())[:max_compares]

                    for db_name, db_encoding in db_items:
                        distance = face_recognition.face_distance([db_encoding], encoding)[0]
                        if distance < min_distance and distance < self.recognition_threshold:
                            min_distance = distance
                            name = db_name

                    results.append((name, (x, y, w, h), min_distance))

            return results
        except Exception as e:
            logging.error(f"快速人脸识别失败: {e}")
            return []

    def get_database_info(self):
        """获取数据库信息"""
        with QMutexLocker(self.db_mutex):
            return {
                "total_faces": len(self.face_database),
                "names": list(self.face_database.keys()),
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }


class RecognitionThread(QThread):
    """人脸识别线程（性能优化）"""
    recognition_complete = pyqtSignal(list)

    def __init__(self, recognizer, frame):
        super().__init__()
        self.recognizer = recognizer
        self.frame = frame.copy()
        self.setTerminationEnabled(True)

    def run(self):
        try:
            results = self.recognizer.recognize_fast(self.frame)
            self.recognition_complete.emit(results)
        except Exception as e:
            logging.error(f"识别线程错误: {e}")
            self.recognition_complete.emit([])
        finally:
            # 释放内存
            del self.frame


class CaptureFaceThread(QThread):
    """人脸采集线程（性能优化）"""
    capture_complete = pyqtSignal(bool, str)
    capture_progress = pyqtSignal(int)

    def __init__(self, recognizer, frame, name):
        super().__init__()
        self.recognizer = recognizer
        self.frame = frame.copy()
        self.name = name.strip()
        self.setTerminationEnabled(True)

    def run(self):
        try:
            self.capture_progress.emit(20)

            # 快速人脸检测
            face_locations = self.recognizer.fast_detect_faces(self.frame)
            self.capture_progress.emit(40)

            if len(face_locations) == 0:
                self.capture_complete.emit(False, "未检测到人脸")
                return
            elif len(face_locations) > 1:
                self.capture_complete.emit(False, f"检测到 {len(face_locations)} 个人脸")
                return

            self.capture_progress.emit(60)

            # 提取特征
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            x, y, w, h = face_locations[0]
            face_recognition_location = [(y, x + w, y + h, x)]

            try:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_recognition_location)
                self.capture_progress.emit(80)
            except Exception as e:
                logging.error(f"人脸特征提取失败: {e}")
                self.capture_complete.emit(False, f"特征提取失败: {str(e)}")
                return

            if not face_encodings:
                self.capture_complete.emit(False, "无法提取人脸特征")
                return

            # 安全保存到数据库
            try:
                self.capture_progress.emit(90)  # 添加中间进度
                
                # 检查数据库文件是否存在，如果不存在则创建
                db_path = self.recognizer.face_database_path
                if not os.path.exists(db_path):
                    # 确保目录存在
                    db_dir = os.path.dirname(db_path)
                    if db_dir and not os.path.exists(db_dir):
                        os.makedirs(db_dir)
                    
                    # 创建空的数据库文件
                    with open(db_path, 'w', encoding='utf-8') as f:
                        json.dump({}, f)
                    logging.info("创建了新的数据库文件")

                # 添加姓名前缀检查
                safe_name = self.name.strip()
                if not safe_name:
                    self.capture_complete.emit(False, "姓名不能为空")
                    return

                # 检查重复姓名 - 临时获取锁检查是否存在
                temp_mutex = QMutexLocker(self.recognizer.db_mutex)
                name_exists = safe_name in self.recognizer.face_database
                temp_mutex.unlock()
                
                if name_exists:
                    # 询问是否覆盖
                    self.capture_complete.emit(False, f"姓名 '{safe_name}' 已存在")
                    return
                
                # 保存到内存数据库 - 获取锁
                with QMutexLocker(self.recognizer.db_mutex):
                    self.recognizer.face_database[safe_name] = face_encodings[0]
                    
                    # 直接保存文件，避免死锁
                    try:
                        data = {name: encoding.tolist() for name, encoding in self.recognizer.face_database.items()}
                        with open(db_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    except Exception as save_e:
                        # 如果保存失败，撤销内存中的数据
                        del self.recognizer.face_database[safe_name]
                        logging.error(f"保存文件失败，已撤销数据: {save_e}")
                        raise save_e
                
                self.capture_progress.emit(100)
                self.capture_complete.emit(True, f"成功采集 {safe_name} 的人脸特征")

            except PermissionError as e:
                logging.error(f"数据库文件权限错误: {e}")
                self.capture_complete.emit(False, "数据库文件权限不足，请检查文件属性")
            except json.JSONDecodeError as e:
                logging.error(f"JSON格式错误: {e}")
                # 备份并重建数据库
                backup_path = f"{self.recognizer.face_database_path}.backup"
                try:
                    if os.path.exists(self.recognizer.face_database_path):
                        import shutil
                        shutil.copy2(self.recognizer.face_database_path, backup_path)
                    
                    # 重建数据库 - 使用临时数据避免死锁
                    temp_data = {safe_name: face_encodings[0].tolist()}
                    with open(self.recognizer.face_database_path, 'w', encoding='utf-8') as f:
                        json.dump(temp_data, f, ensure_ascii=False, indent=2)
                    
                    # 更新内存数据库
                    with QMutexLocker(self.recognizer.db_mutex):
                        self.recognizer.face_database[safe_name] = face_encodings[0]
                    
                    self.capture_progress.emit(100)
                    self.capture_complete.emit(True, f"成功采集 {safe_name} 的人脸特征（数据库已重建）")
                    logging.info(f"数据库已重建，原文件备份为 {backup_path}")
                    
                except Exception as backup_e:
                    logging.error(f"数据库重建失败: {backup_e}")
                    self.capture_complete.emit(False, "数据库文件损坏且修复失败")
            
            except Exception as e:
                logging.error(f"保存数据库失败: {e}")
                self.capture_complete.emit(False, f"保存失败: {str(e)}")

        except Exception as e:
            logging.error(f"采集人脸失败: {e}")
            self.capture_complete.emit(False, f"采集失败: {str(e)}")
        finally:
            # 释放内存
            del self.frame


class FaceRecognitionApp(QMainWindow):
    """主应用程序（性能优化版）"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎭 智能语音播报人脸识别系统 - 流畅版")
        self.setGeometry(100, 100, 1200, 800)  # 缩小窗口尺寸

        # 初始化组件
        self.recognizer = FaceRecognizer()
        self.speaker = VoiceSpeaker()

        # 状态变量
        self.is_recognizing = False
        self.is_camera_available = False
        self.last_speak_time = {}
        self.recognition_results = []
        self.frame_mutex = QMutex()
        self.current_frame = None
        self.current_recognition_thread = None
        self.frame_skip_counter = 0  # 帧跳过计数器
        self.frame_skip_interval = 2  # 每3帧处理1帧

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
        except:
            return self.get_default_config()

    def get_default_config(self):
        """获取默认配置"""
        return {
            "mode": "简洁模式",
            "volume": 0.8,
            "rate": 150,
            "enable_unknown_alert": True,
            "speak_cooldown": 5,
            "camera_index": 0,
            "recognition_threshold": 0.6,
            "performance_mode": "平衡"  # 新增性能模式
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
        self.speaker.set_volume(self.config.get('volume', 0.8))
        self.speaker.set_rate(self.config.get('rate', 150))

        self.speak_cooldown = self.config.get('speak_cooldown', 5)
        self.enable_unknown_alert = self.config.get('enable_unknown_alert', True)
        self.current_mode = self.config.get('mode', '简洁模式')
        self.recognizer.recognition_threshold = self.config.get('recognition_threshold', 0.6)

        # 根据性能模式调整参数
        performance_mode = self.config.get('performance_mode', '平衡')
        if performance_mode == '流畅':
            self.frame_skip_interval = 3
            self.recognizer.recognition_interval = 1.5
        elif performance_mode == '平衡':
            self.frame_skip_interval = 2
            self.recognizer.recognition_interval = 1.0
        else:  # 精准
            self.frame_skip_interval = 1
            self.recognizer.recognition_interval = 0.5

    def init_ui(self):
        """初始化用户界面（简化版）"""
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #2b2b2b; 
                color: white; 
            }
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                border: none;
                padding: 6px 12px; 
                border-radius: 4px; 
                font-size: 12px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
            QLabel { color: white; font-size: 12px; }
            QComboBox, QSlider {
                background-color: #3c3c3c; 
                color: white;
                border: 1px solid #555; 
                border-radius: 4px;
            }
            QGroupBox {
                color: #4CAF50; 
                font-weight: bold;
                border: 1px solid #4CAF50; 
                border-radius: 6px;
                margin-top: 8px; 
                padding-top: 8px;
            }
            QTextEdit {
                background-color: #1e1e1e; 
                color: #00ff00;
                border: 1px solid #555; 
                border-radius: 4px;
                font-family: 'Courier New';
                font-size: 10px;
            }
            QProgressBar {
                height: 8px; 
                border-radius: 4px; 
                background-color: #333333;
            }
            QProgressBar::chunk { 
                background-color: #4CAF50; 
                border-radius: 4px; 
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
        self.video_label.setMinimumSize(640, 480)  # 缩小显示尺寸
        self.video_label.setStyleSheet("""
            QLabel { 
                background: black; 
                color: white; 
                font-size: 14px; 
                border: 1px solid #555; 
            }
        """)

        # 控制按钮
        control_layout = QHBoxLayout()
        self.recognize_btn = QPushButton("开始识别")
        self.capture_btn = QPushButton("采集人脸")
        self.clear_db_btn = QPushButton("清空数据库")
        control_layout.addWidget(self.recognize_btn)
        control_layout.addWidget(self.capture_btn)
        control_layout.addWidget(self.clear_db_btn)

        # 采集进度条
        self.capture_progress_bar = QProgressBar()
        self.capture_progress_bar.setRange(0, 100)
        self.capture_progress_bar.setVisible(False)

        # 状态显示
        self.status_label = QLabel("🔴 系统就绪")
        self.status_label.setStyleSheet("QLabel { font-size: 14px; color: #ff6b6b; }")

        left_layout.addWidget(self.video_label)
        left_layout.addLayout(control_layout)
        left_layout.addWidget(self.capture_progress_bar)
        left_layout.addWidget(self.status_label)

        # 右侧控制面板（简化）
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setAlignment(Qt.AlignTop)

        # 性能设置
        perf_group = QGroupBox("⚡ 性能设置")
        perf_layout = QFormLayout(perf_group)
        self.performance_combo = QComboBox()
        self.performance_combo.addItems(["流畅", "平衡", "精准"])
        self.performance_combo.setCurrentText(self.config.get('performance_mode', '平衡'))
        perf_layout.addRow("性能模式:", self.performance_combo)

        # 语音设置
        voice_group = QGroupBox("🎵 语音设置")
        voice_layout = QFormLayout(voice_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["简洁模式", "静音模式"])
        self.mode_combo.setCurrentText(self.current_mode)
        voice_layout.addRow("播报模式:", self.mode_combo)

        # 识别设置
        recog_group = QGroupBox("🔍 识别设置")
        recog_layout = QFormLayout(recog_group)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(4, 8)  # 0.4-0.8
        self.threshold_slider.setValue(int(self.config.get('recognition_threshold', 0.6) * 10))
        recog_layout.addRow("识别阈值:", self.threshold_slider)

        # 数据库信息
        db_group = QGroupBox("💾 数据库信息")
        db_layout = QVBoxLayout(db_group)
        self.db_info_label = QLabel("加载中...")
        self.db_info_label.setWordWrap(True)
        db_layout.addWidget(self.db_info_label)

        # 系统日志
        log_group = QGroupBox("📝 系统日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        right_layout.addWidget(perf_group)
        right_layout.addWidget(voice_group)
        right_layout.addWidget(recog_group)
        right_layout.addWidget(db_group)
        right_layout.addWidget(log_group)

        # 分割布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 300])
        layout.addWidget(splitter)

        # 连接信号
        self.connect_signals()
        self.update_database_info()

    def connect_signals(self):
        """连接信号槽"""
        self.recognize_btn.clicked.connect(self.toggle_recognition)
        self.capture_btn.clicked.connect(self.capture_face)
        self.clear_db_btn.clicked.connect(self.clear_database)

        self.performance_combo.currentTextChanged.connect(self.on_performance_change)
        self.mode_combo.currentTextChanged.connect(self.on_mode_change)
        self.threshold_slider.valueChanged.connect(self.on_threshold_change)

    def init_camera(self):
        """初始化摄像头（性能优化）"""
        try:
            camera_index = self.config.get('camera_index', 0)
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

            # 降低分辨率提高性能
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # 降低帧率
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小化缓冲区
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 关闭自动对焦

            if self.cap.isOpened():
                self.is_camera_available = True
                self.log_message("✅ 摄像头初始化成功（640x480）")
            else:
                self.is_camera_available = False
                self.log_message("❌ 摄像头初始化失败")

        except Exception as e:
            self.is_camera_available = False
            self.log_message(f"❌ 摄像头初始化错误: {e}")

    def toggle_recognition(self):
        """切换识别状态"""
        if not self.is_camera_available:
            QMessageBox.warning(self, "警告", "摄像头不可用")
            return

        if not self.is_recognizing:
            self.is_recognizing = True
            self.recognize_btn.setText("停止识别")
            self.recognize_btn.setStyleSheet("background-color: #ff6b6b;")
            self.status_label.setText("🟢 识别中...")
            self.timer.start(67)  # ~15 FPS
            self.log_message("🎯 开始人脸识别")
        else:
            self.stop_recognition()

    def stop_recognition(self):
        """停止识别"""
        self.is_recognizing = False
        self.recognize_btn.setText("开始识别")
        self.recognize_btn.setStyleSheet("background-color: #4CAF50;")
        self.status_label.setText("🔴 识别已停止")
        self.timer.stop()

        if self.current_recognition_thread and self.current_recognition_thread.isRunning():
            self.current_recognition_thread.quit()
            self.current_recognition_thread.wait(500)

        self.current_recognition_thread = None
        self.log_message("⏹️ 停止人脸识别")

    def process_frame(self):
        """处理视频帧（性能优化）"""
        if not self.is_camera_available:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # 帧跳过机制，减少处理频率
        self.frame_skip_counter += 1
        if self.frame_skip_counter < self.frame_skip_interval:
            # 只显示，不处理识别
            self.display_frame_fast(frame, self.recognition_results)
            return

        self.frame_skip_counter = 0

        with QMutexLocker(self.frame_mutex):
            self.current_frame = frame.copy()

        # 显示帧
        self.display_frame_fast(frame, self.recognition_results)

        if self.is_recognizing:
            # 确保只有一个识别线程在运行
            if (self.current_recognition_thread is None or
                    not self.current_recognition_thread.isRunning()):
                with QMutexLocker(self.frame_mutex):
                    thread_frame = self.current_frame.copy() if self.current_frame is not None else frame

                self.current_recognition_thread = RecognitionThread(self.recognizer, thread_frame)
                self.current_recognition_thread.recognition_complete.connect(self.on_recognition_complete)
                self.current_recognition_thread.start()

    def on_recognition_complete(self, results):
        """识别完成回调"""
        self.recognition_results = results
        self.handle_voice_announce(results)

    def display_frame_fast(self, frame, results=None):
        """快速显示视频帧（性能优化）"""
        try:
            display_frame = frame.copy()

            if results:
                # 简化绘制逻辑
                for name, bbox, confidence in results:
                    x, y, w, h = bbox
                    color = (0, 0, 255) if name == "未知人员" else (0, 255, 0)

                    # 只绘制矩形，不绘制文本（减少计算）
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                    # 只在精准模式下绘制文本
                    if self.config.get('performance_mode', '平衡') == '精准':
                        label = f"{name}"
                        cv2.putText(display_frame, label, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 快速转换图像格式
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            # 使用更快的图像缩放
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            if not qt_image.isNull():
                # 使用快速缩放
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.video_label.width(),
                    self.video_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation  # 使用快速变换
                )
                self.video_label.setPixmap(pixmap)

        except Exception as e:
            # 简化错误处理
            pass

    def capture_face(self):
        """采集人脸"""
        if not self.is_camera_available:
            QMessageBox.warning(self, "警告", "摄像头不可用")
            return

        # 停止识别以确保稳定性
        if self.is_recognizing:
            self.stop_recognition()
            QTimer.singleShot(300, self._do_capture)
        else:
            self._do_capture()

    def _do_capture(self):
        """执行采集操作"""
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "错误", "无法获取摄像头画面")
            return

        frame = cv2.flip(frame, 1)

        # 快速检测人脸
        face_locations = self.recognizer.fast_detect_faces(frame)

        if len(face_locations) == 0:
            QMessageBox.warning(self, "提示", "未检测到人脸")
            return
        elif len(face_locations) > 1:
            QMessageBox.warning(self, "提示", "检测到多个人脸")
            return

        name, ok = QInputDialog.getText(self, "采集人脸", "请输入姓名:")
        if ok and name.strip():
            self.capture_progress_bar.setVisible(True)
            self.capture_progress_bar.setValue(0)

            self.capture_thread = CaptureFaceThread(self.recognizer, frame, name)
            self.capture_thread.capture_complete.connect(self.on_capture_complete)
            self.capture_thread.capture_progress.connect(self.on_capture_progress)
            self.capture_thread.start()

    def on_capture_progress(self, value):
        """更新采集进度"""
        self.capture_progress_bar.setValue(value)

    def on_capture_complete(self, success, message):
        """采集完成回调"""
        self.capture_progress_bar.setVisible(False)
        self.capture_progress_bar.setValue(0)

        if success:
            self.log_message(f"✅ {message}")
            self.update_database_info()
            # 简化成功提示
            QMessageBox.information(self, "成功", message)
        else:
            self.log_message(f"❌ {message}")
            QMessageBox.warning(self, "失败", message)

    def handle_voice_announce(self, results):
        """处理语音播报"""
        if not results or self.current_mode == "静音模式":
            return

        current_time = time.time()
        names_to_speak = []

        for name, _, confidence in results:
            last_time = self.last_speak_time.get(name, 0)
            if current_time - last_time > self.speak_cooldown:
                names_to_speak.append(name)
                self.last_speak_time[name] = current_time

        if names_to_speak:
            self.speaker.speak_face_result(names_to_speak, self.current_mode, self.enable_unknown_alert)

    def clear_database(self):
        """清空数据库"""
        reply = QMessageBox.question(self, "确认", "确定要清空所有人脸数据吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.recognizer.face_database = {}
            self.recognizer.save_face_database()
            self.update_database_info()
            self.log_message("🗑️ 人脸数据库已清空")

    def update_database_info(self):
        """更新数据库信息"""
        db_info = self.recognizer.get_database_info()
        info_text = f"总人数: {db_info['total_faces']}\n"
        if db_info['names']:
            info_text += f"名单: {', '.join(db_info['names'][:3])}"  # 只显示前3个
            if len(db_info['names']) > 3:
                info_text += "..."
        self.db_info_label.setText(info_text)

    def log_message(self, message):
        """记录日志（简化）"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        # 限制日志长度
        if self.log_text.document().lineCount() > 50:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 10)
            cursor.removeSelectedText()

    def on_performance_change(self, mode):
        """性能模式改变"""
        self.config['performance_mode'] = mode
        self.save_config()
        self.apply_config()
        self.log_message(f"⚡ 性能模式: {mode}")

    def on_mode_change(self, mode):
        self.current_mode = mode
        self.config['mode'] = mode
        self.save_config()

    def on_threshold_change(self, value):
        threshold = value / 10.0
        self.recognizer.recognition_threshold = threshold
        self.config['recognition_threshold'] = threshold
        self.save_config()

    def closeEvent(self, event):
        """安全关闭应用程序"""
        self.stop_recognition()

        if hasattr(self, 'cap'):
            self.cap.release()

        if hasattr(self, 'speaker'):
            self.speaker.stop()

        self.save_config()
        event.accept()


if __name__ == "__main__":
    # 设置高性能模式
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("人脸识别系统 - 流畅版")


    # 简化异常处理
    def exception_handler(exctype, value, traceback):
        logging.error(f"异常: {exctype.__name__}: {value}")


    sys.excepthook = exception_handler

    window = FaceRecognitionApp()
    window.show()

    try:
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"应用程序退出: {e}")


