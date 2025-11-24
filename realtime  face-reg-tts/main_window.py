# 导入必要的库
# -*- coding: utf-8 -*-
"""
main_window.py
实时人脸识别与语音播报系统的主窗口模块

该模块实现了系统的图形用户界面，包括视频显示、人脸识别结果展示、
人脸注册管理、系统状态监控和日志显示等功能。
"""

# 标准库导入
import sys  # 提供系统相关功能，如退出程序
import time  # 时间模块，用于记录时间戳和计算运行时间

# 第三方库导入
import cv2  # OpenCV库，用于摄像头操作、图像处理和人脸检测
import numpy as np  # 数值计算库，用于图像处理相关计算
import face_recognition  # 人脸识别库，提供面部特征提取和匹配功能

# PyQt5组件导入，用于构建图形界面
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QVBoxLayout, QHBoxLayout,  # 窗口和布局类
    QWidget, QLabel, QPushButton, QTextEdit, QLineEdit,  # 基本控件
    QFileDialog, QMessageBox, QSplitter, QFrame, QGroupBox,  # 对话框和容器
    QGridLayout, QComboBox, QListWidget, QListWidgetItem,  # 高级布局和列表
    QDialog, QFormLayout, QScrollArea, QProgressBar  # 对话框和滚动区域
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon  # 图像和字体处理
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QObject  # Qt核心功能

# 自定义模块导入
from logger import logger  # 日志记录器，用于系统日志记录
from tts_engine import tts_engine  # 文本转语音引擎，提供语音播报功能
from face_database import face_db  # 人脸数据库，用于存储和管理已注册人脸
from face_recognition_system import FaceRecognitionSystem  # 人脸识别系统，处理人脸识别核心逻辑
from face_system_observer import FaceSystemObserver  # 人脸识别系统观察者，用于事件通知机制


class VideoThread(QThread):
    """
    视频处理线程类
    
    该类继承自QThread，在独立线程中负责从摄像头捕获视频帧、处理人脸识别
    并通过信号机制将结果传递给主线程。采用生产者-消费者模式设计，确保
    图像处理不会阻塞UI响应。
    
    主要功能：
    - 摄像头视频帧捕获与处理
    - 人脸检测与识别
    - 实时状态监控与更新
    - 通过信号机制与主线程安全通信
    """
    # 定义信号，用于发送图像到主线程进行显示
    # 参数: OpenCV格式的图像数组 (np.ndarray)
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    # 定义信号，用于发送人脸识别结果
    # 参数: 识别结果列表，每项为(name, confidence)元组
    recognition_result_signal = pyqtSignal(list)
    
    # 定义信号，用于发送系统状态信息
    # 参数: (status_text: 状态文本, status_color: 状态颜色)
    status_signal = pyqtSignal(str, str)  
    
    def __init__(self, camera_id=0):
        """
        初始化视频处理线程
        
        Args:
            camera_id: 摄像头ID，默认为0（系统默认摄像头）
        """
        super().__init__()
        # 线程配置参数
        self.camera_id = camera_id  # 摄像头设备ID
        self.running = False  # 线程运行状态标志
        
        # 处理相关属性
        self.face_system = None  # 人脸识别系统实例
        self.frame_count = 0  # 总帧数计数器
        self.last_process_time = 0  # 上次处理时间戳，用于计算FPS
        self.process_interval = 2  # 帧处理间隔，控制人脸识别频率，降低CPU占用
    
    def run(self):
        """
        线程主函数
        
        线程执行的核心逻辑，负责：
        1. 初始化摄像头设备
        2. 创建人脸识别系统实例
        3. 加载人脸数据库
        4. 持续捕获视频帧并进行处理
        5. 计算和显示FPS
        6. 发送状态更新信号
        7. 释放资源
        """
        self.running = True  # 启动标志设为True
        
        # 创建并初始化摄像头对象
        cap = cv2.VideoCapture(self.camera_id)
        
        # 创建人脸识别系统实例，处理人脸检测和识别核心逻辑
        self.face_system = FaceRecognitionSystem()
        
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            logger.error(f"无法打开摄像头，ID: {self.camera_id}")
            self.status_signal.emit("摄像头错误", "red")
            return
        
        logger.info(f"摄像头启动成功，ID: {self.camera_id}")
        self.status_signal.emit("运行中", "green")
        
        # 从目录初始化人脸数据库
        face_db.initialize_database_from_directory()
        
        while self.running:
            # 从摄像头读取单帧图像
            ret, frame = cap.read()
            if not ret:
                # 处理视频读取失败情况
                logger.error("无法读取视频帧")
                self.status_signal.emit("视频读取错误", "red")
                break
            
            # 增加帧计数，用于帧率计算和处理间隔控制
            self.frame_count += 1
            
            # 每隔指定帧数处理一次人脸，降低计算负担
            if self.frame_count % self.process_interval == 0:
                # 创建帧的副本进行处理，避免原图像被修改
                recognized_faces = self._process_faces(frame.copy())
                # 如有识别结果，发送到主线程
                if recognized_faces:
                    self.recognition_result_signal.emit(recognized_faces)
            
            # 计算并显示FPS (每秒帧数)
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_process_time) if self.last_process_time else 0
            self.last_process_time = current_time
            
            # 在视频帧上绘制FPS信息，便于实时监控性能
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 获取并显示已注册人脸数量
            face_count = len(face_db.get_all_faces())
            cv2.putText(frame, f"已注册: {face_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 发送处理后的图像到主线程进行显示
            self.change_pixmap_signal.emit(frame)
            
            # 短暂休眠，控制帧率，避免CPU占用过高
            time.sleep(0.01)
        
        # 释放摄像头资源
        cap.release()
        self.status_signal.emit("已停止", "gray")
    
    def _process_faces(self, frame):
        """
        处理视频帧中的人脸
        
        核心人脸识别算法实现，步骤包括：
        1. 加载已知人脸数据
        2. 人脸检测（定位人脸位置）
        3. 特征提取（计算人脸编码）
        4. 人脸匹配（计算与已知人脸的相似度）
        5. 结果处理与可视化
        
        Args:
            frame: 要处理的视频帧（BGR格式）
            
        Returns:
            list: 识别到的人脸列表，每项为(name, confidence)元组
        """
        try:
            # 从人脸数据库获取所有已注册人脸信息
            all_faces = face_db.get_all_faces()
            # 如果数据库为空，直接返回空列表
            if not all_faces:
                return []
            
            # 解包人脸编码和对应的名称，准备匹配
            known_face_encodings = [encoding for _, encoding in all_faces]
            known_face_names = [name for name, _ in all_faces]
            
            # 颜色空间转换：OpenCV默认BGR，face_recognition库需要RGB格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 人脸检测：获取图像中所有人脸的位置坐标
            face_locations = face_recognition.face_locations(rgb_frame)
            # 人脸特征提取：计算每个人脸的128维特征向量
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # 存储识别结果的列表
            recognized_faces = []
            
            # 遍历每一个检测到的人脸和其特征向量
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # 计算当前人脸与已知人脸的欧氏距离（相似度）
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                # 找到距离最小的人脸（最相似的匹配）
                best_match_index = np.argmin(face_distances)
                best_match_distance = face_distances[best_match_index]
                
                # 置信度阈值，距离小于此值则认为匹配成功
                tolerance = 0.6
                
                if best_match_distance < tolerance:
                    # 匹配成功：获取对应的人名和计算置信度
                    name = known_face_names[best_match_index]
                    confidence = 1.0 - best_match_distance  # 距离越小，置信度越高
                    recognized_faces.append((name, confidence))
                    
                    # 可视化：绘制绿色矩形框标记已识别的人脸
                    color = (0, 255, 0)  # 绿色
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # 绘制包含姓名和置信度的标签
                    label = f"{name} ({confidence:.2f})"
                    # 标签背景
                    cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                    # 标签文字
                    cv2.putText(frame, label, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    # 匹配失败：标记为未知人脸
                    color = (0, 0, 255)  # 红色
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    # 绘制"未知"标签
                    cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
                    cv2.putText(frame, "未知", (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 返回所有成功识别的人脸信息
            return recognized_faces
            
        except Exception as e:
            # 异常处理：记录错误日志并返回空列表
            logger.error(f"处理人脸时出错: {str(e)}")
            return []
    
    def stop(self):
        """
        停止视频线程
        
        设置运行标志为False，并等待线程安全退出，确保资源正确释放
        """
        self.running = False  # 设置停止标志
        self.wait()  # 等待线程完成当前操作后退出





class MainWindow(QMainWindow):
    """
    实时人脸识别系统的主窗口类
    
    该类继承自QMainWindow，是整个应用程序的核心界面组件，负责：
    - 视频显示和人脸识别结果可视化
    - 人脸注册与管理功能
    - 系统状态监控与日志显示
    - 用户交互与事件响应
    - 线程管理与信号处理
    
    采用MVC设计模式，作为视图层与控制系统的核心交互界面。
    """
    
    def __init__(self, config=None, face_db=None, tts_engine=None, face_system=None):
        """
        初始化主窗口
        
        Args:
            config: 系统配置对象（可选）
            face_db: 人脸数据库实例（可选）
            tts_engine: 文本转语音引擎实例（可选）
            face_system: 人脸识别系统实例（可选）
        """
        super().__init__()
        
        # 初始化系统核心组件
        self.config = config
        self.face_db = face_db or face_db  # 人脸数据库引用
        self.tts_engine = tts_engine or tts_engine  # 文本转语音引擎引用
        self.face_system = face_system or face_system  # 人脸识别系统引用
        
        # 实现观察者模式，用于接收人脸识别系统事件通知
        self.observer = FaceSystemObserver()
        
        # 安全检查并注册观察者
        if self.face_system:
            self.face_system.register_observer(self.observer)
            
        # 初始化视频线程
        self.video_thread = None  # 视频处理线程引用
        
        # 初始化用户界面
        self.init_ui()  # 创建和布局所有UI组件
    
    def init_ui(self):
        """
        初始化用户界面
        设置窗口属性、创建布局和控件
        """
        # 设置窗口标题和大小
        self.setWindowTitle("实时人脸识别与语音播报系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化识别相关变量
        self.last_recognized_name = None
        self.last_recognized_time = None
        
        # 连接信号和槽
        self.connect_signals_slots()
        
        # 创建菜单栏
        self.create_menus()
        
        # 创建工具栏
        self.create_toolbars()
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建左侧布局（视频显示区域）
        left_layout = QVBoxLayout()
        
        # 创建视频显示标签
        self.video_label = QLabel("摄像头预览")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid #ccc;")
        left_layout.addWidget(self.video_label)
        
        # 创建控制面板
        control_group = QGroupBox("系统控制")
        control_layout = QHBoxLayout()
        
        # 开始按钮
        self.start_button = QPushButton("开始识别")
        self.start_button.clicked.connect(self.start_recognition)
        control_layout.addWidget(self.start_button)
        
        # 停止按钮
        self.stop_button = QPushButton("停止识别")
        self.stop_button.clicked.connect(self.stop_recognition)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        # 摄像头选择
        camera_layout = QHBoxLayout()
        camera_label = QLabel("摄像头:")
        self.camera_combo = QComboBox()
        # 添加常见的摄像头ID选项
        for i in range(3):
            self.camera_combo.addItem(f"摄像头 {i}", i)
        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_combo)
        control_layout.addLayout(camera_layout)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧窗口
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)
        
        # 创建右侧布局
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 创建注册面板
        register_group = QGroupBox("人脸注册")
        register_layout = QVBoxLayout()
        
        # 显示已注册人员列表
        self.face_list_widget = QListWidget()
        self.face_list_widget.setAlternatingRowColors(True)
        self.face_list_widget.setMinimumHeight(100)
        
        # 刷新已注册列表按钮
        refresh_layout = QHBoxLayout()
        self.refresh_list_button = QPushButton("刷新列表")
        self.refresh_list_button.clicked.connect(self.refresh_face_list)
        refresh_layout.addWidget(self.refresh_list_button)
        
        self.delete_face_button = QPushButton("删除选中")
        self.delete_face_button.clicked.connect(self.delete_selected_face)
        refresh_layout.addWidget(self.delete_face_button)
        
        # 详细注册按钮
        self.advanced_register_button = QPushButton("高级注册...")
        self.advanced_register_button.clicked.connect(self.show_advanced_registration)
        refresh_layout.addWidget(self.advanced_register_button)
        
        register_layout.addWidget(QLabel("已注册人员:"))
        register_layout.addWidget(self.face_list_widget)
        register_layout.addLayout(refresh_layout)
        
        # 快速注册区域
        quick_register_group = QGroupBox("快速注册")
        quick_register_layout = QVBoxLayout()
        
        # 姓名输入
        name_layout = QHBoxLayout()
        name_label = QLabel("姓名:")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入姓名")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        quick_register_layout.addLayout(name_layout)
        
        # 快速注册按钮布局
        quick_register_buttons_layout = QHBoxLayout()
        self.register_camera_button = QPushButton("从摄像头注册")
        self.register_camera_button.clicked.connect(self.register_from_camera)
        quick_register_buttons_layout.addWidget(self.register_camera_button)
        
        self.register_file_button = QPushButton("从文件注册")
        self.register_file_button.clicked.connect(self.register_from_file)
        quick_register_buttons_layout.addWidget(self.register_file_button)
        quick_register_layout.addLayout(quick_register_buttons_layout)
        
        quick_register_group.setLayout(quick_register_layout)
        register_layout.addWidget(quick_register_group)
        
        register_group.setLayout(register_layout)
        right_layout.addWidget(register_group)
        
        # 初始化时刷新人脸列表
        self.refresh_face_list()
        
        # 创建识别结果面板
        result_group = QGroupBox("最近识别结果")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        # 创建日志面板
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        # 系统状态统计变量
        self.start_time = None
        self.total_frames = 0
        self.total_recognitions = 0
        self.fps = 0.0
        self.last_fps_update = 0
        self.frame_count_since_last_fps = 0
        
        # 创建状态栏 - 增强版，显示更多系统信息
        status_group = QGroupBox("系统状态")
        status_layout = QGridLayout()  # 使用网格布局显示更多信息
        
        # 状态信息
        status_layout.addWidget(QLabel("运行状态:"), 0, 0)
        self.status_text = QLabel("就绪")
        self.status_text.setStyleSheet("color: gray;")
        status_layout.addWidget(self.status_text, 0, 1)
        
        # FPS信息
        status_layout.addWidget(QLabel("FPS:"), 1, 0)
        self.fps_label = QLabel("0.0")
        status_layout.addWidget(self.fps_label, 1, 1)
        
        # 运行时间
        status_layout.addWidget(QLabel("运行时间:"), 0, 2)
        self.runtime_label = QLabel("00:00:00")
        status_layout.addWidget(self.runtime_label, 0, 3)
        
        # 识别次数
        status_layout.addWidget(QLabel("识别次数:"), 1, 2)
        self.recognitions_label = QLabel("0")
        status_layout.addWidget(self.recognitions_label, 1, 3)
        
        # 已注册人数
        status_layout.addWidget(QLabel("已注册人数:"), 0, 4)
        self.registered_count_label = QLabel("0")
        status_layout.addWidget(self.registered_count_label, 0, 5)
        
        # 添加拉伸因子
        status_layout.setColumnStretch(6, 1)
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)
        
        # 设置右侧布局的拉伸因子
        right_layout.setStretch(0, 1)  # 注册面板
        right_layout.setStretch(1, 2)  # 识别结果
        right_layout.setStretch(2, 3)  # 日志
        right_layout.setStretch(3, 1)  # 状态
        
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        splitter.setSizes([800, 400])
        
        # 添加分割器到主布局
        main_layout.addWidget(splitter)
        
        # 创建状态栏
        self.statusBar().showMessage("就绪")
        
        # 创建定时器，用于更新日志显示
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self.update_log_display)
        self.log_timer.start(1000)  # 每秒更新一次
        
        # 创建状态更新计时器
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_system_status)
        self.status_timer.start(100)  # 每100毫秒更新一次状态信息
        
        # 设置全局快捷键
        self.setup_shortcuts()
    
    def connect_signals_slots(self):
        """
        连接信号和槽函数
        """
        # 连接观察者信号
        if hasattr(self, 'observer'):
            self.observer.frame_processed_signal.connect(self.update_image)
            self.observer.face_recognized_signal.connect(self.handle_face_recognition)
            self.observer.system_status_changed_signal.connect(self.handle_system_status_change)
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """
        更新视频显示图像
        
        Args:
            cv_img: OpenCV格式的图像
        """
        # 增加帧数计数
        self.increment_frame_count()
        
        # 转换OpenCV图像为Qt图像
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 调整图像大小以适应标签
        scaled_image = qt_image.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 显示图像
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))
    
    @pyqtSlot(list)
    def update_recognition_result(self, results):
        """
        更新识别结果显示
        
        Args:
            results: 识别结果列表 [(name, confidence), ...]
        """
        for name, confidence in results:
            # 处理识别结果
            self.handle_face_recognition(name, confidence)
            
            # 添加到识别结果文本框
            timestamp = time.strftime("%H:%M:%S")
            result_text = f"[{timestamp}] 识别到: {name} (置信度: {confidence:.2f})"
            self.result_text.append(result_text)
            
            # 限制文本框行数
            lines = self.result_text.toPlainText().split('\n')
            if len(lines) > 50:
                self.result_text.setPlainText('\n'.join(lines[-50:]))
            
            # 滚动到底部
            self.result_text.moveCursor(self.result_text.textCursor().End)
    
    def handle_system_status_change(self, is_running):
        """
        处理系统状态变化
        
        Args:
            is_running: 系统是否正在运行
        """
        # 更新UI状态
        self.update_ui_state(is_running)
        
        # 更新日志
        logger.info(f"人脸识别系统状态已变更为: {'运行中' if is_running else '已停止'}")
    
    def handle_face_recognition(self, name, confidence):
        """
        处理人脸识别结果
        
        Args:
            name: 识别到的姓名
            confidence: 置信度
        """
        # 增加识别次数
        self.total_recognitions += 1
        self.recognitions_label.setText(str(self.total_recognitions))
        
        # 更新最近识别信息
        self.last_recognized_name = name
        self.last_recognized_time = time.strftime("%H:%M:%S")
        
        # 更新状态栏
        status = f"识别状态: 运行中 | 最近识别: {name or '无'}"
        self.statusBar().showMessage(status)
        
        # 更新结果显示
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        result_text = f"[{timestamp}] 识别结果: {name}, 置信度: {confidence:.2f}\n"
        self.result_text.insertPlainText(result_text)
        self.result_text.moveCursor(self.result_text.textCursor().End)
        
        # 记录日志
        self.log_message(f"识别到: {name}, 置信度: {confidence:.2f}")
        
        # 通过TTS播报
        if self.tts_engine and name != "Unknown":
            self.tts_engine.speak(f"你好，{name}")
        elif self.tts_engine:
            self.tts_engine.speak("检测到未知人员")
    
    def log_message(self, message):
        """
        记录消息到UI日志区域
        
        Args:
            message: 要记录的消息
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_text = f"[{timestamp}] {message}\n"
        self.log_text.insertPlainText(log_text)
        self.log_text.moveCursor(self.log_text.textCursor().End)
        
        # 同时记录到系统日志
        logger.info(message)
    
    @pyqtSlot(str, str)
    def update_status(self, status, color):
        """
        更新状态栏显示
        
        Args:
            status: 状态文本
            color: 状态颜色
        """
        self.status_text.setText(status)
        self.status_text.setStyleSheet(f"color: {color};")
        self.statusBar().showMessage(status)
        
        # 根据状态更新开始/停止计时
        if status == "识别中" and self.start_time is None:
            self.start_time = time.time()
        elif status == "就绪" or status == "已停止":
            self.start_time = None
            self.fps = 0.0
            self.total_frames = 0
            self.fps_label.setText("0.0")
            self.runtime_label.setText("00:00:00")
            self.recognitions_label.setText("0")
    
    def update_system_status(self):
        """
        更新系统状态信息，包括FPS、运行时间等
        此方法由status_timer定时调用
        """
        current_time = time.time()
        
        # 更新FPS计算
        if self.frame_count_since_last_fps % 10 == 0:  # 每10帧更新一次FPS
            if self.last_fps_update > 0:
                elapsed = current_time - self.last_fps_update
                if elapsed > 0:
                    self.fps = self.frame_count_since_last_fps / elapsed
                    self.fps_label.setText(f"{self.fps:.1f}")
            self.last_fps_update = current_time
            self.frame_count_since_last_fps = 0
        
        # 更新运行时间
        if self.start_time is not None:
            runtime = int(current_time - self.start_time)
            hours, remainder = divmod(runtime, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.runtime_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def increment_frame_count(self):
        """
        增加处理的帧数计数，用于FPS计算
        """
        self.total_frames += 1
        self.frame_count_since_last_fps += 1
    
    def start_recognition(self):
        """开始人脸识别"""
        try:
            # 检查是否已在运行
            if self.face_system and self.face_system.is_running:
                QMessageBox.warning(self, "警告", "识别系统已经在运行中")
                return
            
            # 获取选择的摄像头
            camera_id = self.camera_combo.currentData()
            
            # 使用face_system方式启动
            if self.face_system:
                # 更新人脸识别系统的摄像头ID
                self.face_system.set_camera_id(camera_id)
                
                # 启动人脸识别系统
                self.face_system.start()
            else:
                # 兼容原方式
                # 检查是否已经有线程在运行
                if self.video_thread and self.video_thread.isRunning():
                    QMessageBox.warning(self, "警告", "识别系统已经在运行中")
                    return
                
                # 创建并启动视频线程
                self.video_thread = VideoThread(camera_id)
                
                # 连接信号和槽
                self.video_thread.change_pixmap_signal.connect(self.update_image)
                self.video_thread.recognition_result_signal.connect(self.update_recognition_result)
                self.video_thread.status_signal.connect(self.update_status)
                
                # 启动线程
                self.video_thread.start()
            
            # 更新界面控件状态
            self.update_ui_state(True)
            
            # 更新日志
            logger.info(f"已启动人脸识别系统，使用摄像头 {camera_id}")
            
            # 播报启动信息
            if hasattr(self, 'tts_engine') and self.tts_engine:
                self.tts_engine.speak("人脸识别系统已启动")
            else:
                tts_engine.speak("人脸识别系统已启动")
        except Exception as e:
            logger.error(f"启动识别系统失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"启动识别系统失败: {str(e)}")
            self.update_ui_state(False)
    
    def stop_recognition(self):
        """停止人脸识别"""
        try:
            # 使用face_system方式停止
            if self.face_system and self.face_system.is_running:
                # 显示停止过程状态
                self.statusBar().showMessage("正在停止识别系统...")
                
                # 停止人脸识别系统
                self.face_system.stop()
            else:
                # 兼容原方式
                if self.video_thread:
                    # 显示停止过程状态
                    self.statusBar().showMessage("正在停止识别系统...")
                    
                    # 停止线程
                    self.video_thread.stop()
                    self.video_thread = None
            
            # 更新界面控件状态
            self.update_ui_state(False)
            
            # 更新日志
            logger.info("停止人脸识别")
            
            # 清空视频显示
            self.video_label.setText("摄像头预览")
            
            # 播报停止信息
            if hasattr(self, 'tts_engine') and self.tts_engine:
                self.tts_engine.speak("人脸识别系统已停止")
            else:
                tts_engine.speak("人脸识别系统已停止")
        except Exception as e:
            logger.error(f"停止识别系统失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"停止识别系统失败: {str(e)}")
    
    def refresh_face_list(self):
        """刷新已注册人脸列表并更新已注册人数统计"""
        try:
            # 清空当前列表
            self.face_list_widget.clear()
            
            # 从数据库获取所有人脸
            all_faces = face_db.get_all_faces()
            
            # 添加到列表
            for name, _ in all_faces:
                item = QListWidgetItem(name)
                self.face_list_widget.addItem(item)
            
            # 更新已注册人数显示
            self.registered_count_label.setText(str(len(all_faces)))
            
            # 更新日志
            logger.info(f"已刷新人脸列表，共 {len(all_faces)} 条记录")
            
        except Exception as e:
            logger.error(f"刷新人脸列表失败: {str(e)}")
            QMessageBox.warning(self, "警告", f"刷新人脸列表失败: {str(e)}")
    
    def delete_selected_face(self):
        """删除选中的人脸"""
        # 获取选中的项
        selected_items = self.face_list_widget.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要删除的人员")
            return
        
        # 询问确认
        name = selected_items[0].text()
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除 '{name}' 的人脸信息吗？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # 从数据库删除
                face_db.delete_face(name)
                face_db.save_encodings()
                
                # 刷新列表
                self.refresh_face_list()
                
                # 更新日志
                logger.info(f"已删除人脸: {name}")
                
                # 显示成功信息
                QMessageBox.information(self, "成功", f"已删除 '{name}' 的人脸信息")
                
                # 播报删除信息
                tts_engine.speak(f"已删除 {name} 的人脸信息")
                
            except Exception as e:
                logger.error(f"删除人脸失败: {str(e)}")
                QMessageBox.critical(self, "错误", f"删除失败: {str(e)}")
    
    def show_advanced_registration(self):
        """显示高级注册对话框"""
        # 创建并显示高级注册对话框
        dialog = AdvancedRegistrationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # 注册成功，刷新列表
            self.refresh_face_list()
    
    def register_from_camera(self):
        """从摄像头注册人脸"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入姓名")
            return
        
        # 创建一个临时的摄像头捕获
        cap = cv2.VideoCapture(self.camera_combo.currentData())
        if not cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            return
        
        # 捕获几帧以确保摄像头稳定
        for _ in range(5):
            ret, _ = cap.read()
        
        # 捕获人脸
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            QMessageBox.critical(self, "错误", "无法捕获图像")
            return
        
        # 处理捕获的图像
        try:
            # 转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            face_locations = face_recognition.face_locations(rgb_frame)
            if not face_locations:
                QMessageBox.warning(self, "警告", "未检测到人脸")
                return
            
            # 提取人脸编码
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if not face_encodings:
                QMessageBox.warning(self, "警告", "无法提取人脸特征")
                return
            
            # 添加到数据库
            face_db.add_face(name, face_encodings[0])
            face_db.save_encodings()
            
            # 显示成功消息
            QMessageBox.information(self, "成功", f"已成功注册人脸: {name}")
            logger.info(f"成功注册人脸: {name}")
            
            # 播报成功信息
            tts_engine.speak(f"已成功注册 {name} 的人脸信息")
            
        except Exception as e:
            logger.error(f"注册人脸时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"注册失败: {str(e)}")
    
    def register_from_file(self):
        """从文件注册人脸"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入姓名")
            return
        
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择人脸图片", "", "图片文件 (*.jpg *.jpeg *.png)"
        )
        
        if not file_path:
            return
        
        try:
            # 加载图像
            image = face_recognition.load_image_file(file_path)
            
            # 检测人脸
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                QMessageBox.warning(self, "警告", "图片中未检测到人脸")
                return
            
            # 提取人脸编码
            face_encodings = face_recognition.face_encodings(image, face_locations)
            if not face_encodings:
                QMessageBox.warning(self, "警告", "无法提取人脸特征")
                return
            
            # 添加到数据库
            face_db.add_face(name, face_encodings[0])
            face_db.save_encodings()
            
            # 显示成功消息
            QMessageBox.information(self, "成功", f"已成功注册人脸: {name}")
            logger.info(f"从文件成功注册人脸: {name}")
            
            # 播报成功信息
            tts_engine.speak(f"已成功从文件注册 {name} 的人脸信息")
            
        except Exception as e:
            logger.error(f"从文件注册人脸时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"注册失败: {str(e)}")
    
    def update_log_display(self):
        """
        更新日志显示
        由于我们现在使用log_message方法实时更新日志，此方法主要确保日志区域始终显示最新内容
        """
        # 确保日志区域总是滚动到底部，显示最新的日志内容
        self.log_text.moveCursor(self.log_text.textCursor().End)
    
    def create_menus(self):
        """创建菜单栏"""
        # 创建菜单栏
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        # 开始识别动作
        start_action = file_menu.addAction("开始识别")
        start_action.triggered.connect(self.start_recognition)
        start_action.setShortcut("Ctrl+S")
        
        # 停止识别动作
        stop_action = file_menu.addAction("停止识别")
        stop_action.triggered.connect(self.stop_recognition)
        stop_action.setShortcut("Ctrl+P")
        
        # 添加分隔线
        file_menu.addSeparator()
        
        # 退出动作
        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.confirm_exit)
        exit_action.setShortcut("Ctrl+Q")
        
        # 设置菜单
        setting_menu = menubar.addMenu("设置")
        
        # 人脸识别设置动作
        face_recog_action = setting_menu.addAction("人脸识别设置")
        face_recog_action.triggered.connect(self.show_face_settings)
        
        # TTS设置动作
        tts_action = setting_menu.addAction("语音设置")
        tts_action.triggered.connect(self.show_tts_settings)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        # 关于动作
        about_action = help_menu.addAction("关于")
        about_action.triggered.connect(self.show_about)
    
    def create_toolbars(self):
        """创建工具栏"""
        # 创建主工具栏
        main_toolbar = self.addToolBar("主工具栏")
        
        # 添加开始按钮
        start_action = main_toolbar.addAction("开始识别")
        start_action.triggered.connect(self.start_recognition)
        
        # 添加停止按钮
        stop_action = main_toolbar.addAction("停止识别")
        stop_action.triggered.connect(self.stop_recognition)
        
        # 添加分隔符
        main_toolbar.addSeparator()
        
        # 添加注册按钮
        register_action = main_toolbar.addAction("人脸注册")
        register_action.triggered.connect(self.show_registration_dialog)
    
    def setup_shortcuts(self):
        """设置全局快捷键"""
        pass  # 目前使用菜单的快捷键，如需额外快捷键可在此添加
    
    def update_ui_state(self, is_running):
        """
        更新UI控件状态
        
        Args:
            is_running: 系统是否正在运行
        """
        # 更新按钮状态
        self.start_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)
        
        # 更新菜单栏状态
        menu_bar = self.menuBar()
        if menu_bar.actions():
            file_menu = menu_bar.actions()[0].menu()
            if file_menu and len(file_menu.actions()) >= 2:
                file_menu.actions()[0].setEnabled(not is_running)  # 开始识别
                file_menu.actions()[1].setEnabled(is_running)      # 停止识别
        
        # 更新工具栏状态
        for toolbar in self.findChildren(QPushButton):
            if toolbar.text() == "开始识别":
                toolbar.setEnabled(not is_running)
            elif toolbar.text() == "停止识别":
                toolbar.setEnabled(is_running)
    
    def confirm_exit(self):
        """确认退出系统"""
        # 显示确认对话框
        reply = QMessageBox.question(
            self, "确认退出", 
            "您确定要退出系统吗？\n正在进行的识别任务将被停止。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 停止系统
            self.stop_recognition()
            
            # 停止TTS引擎
            tts_engine.stop()
            
            # 更新日志
            logger.info("用户确认退出系统")
            
            # 退出应用
            QApplication.quit()
    
    def show_face_settings(self):
        """显示人脸识别设置对话框"""
        QMessageBox.information(self, "人脸识别设置", "人脸识别设置功能即将推出")
    
    def show_tts_settings(self):
        """显示TTS设置对话框"""
        QMessageBox.information(self, "语音设置", "语音设置功能即将推出")
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, "关于", 
            "实时人脸识别与语音播报系统 v1.0\n\n" +
            "基于PyQt5开发的图形界面\n" +
            "功能：实时人脸检测、识别和语音播报\n\n" +
            "© 2024 人脸识别系统"
        )
    
    def show_registration_dialog(self):
        """显示人脸注册对话框"""
        # 聚焦到姓名输入框
        self.name_input.setFocus()


class AdvancedRegistrationDialog(QDialog):
    """
    高级人脸注册对话框
    提供多图注册、预览和详细设置功能
    """
    
    def __init__(self, parent=None):
        """初始化高级注册对话框"""
        super().__init__(parent)
        self.setWindowTitle("高级人脸注册")
        self.setMinimumSize(600, 500)
        
        # 存储临时人脸编码
        self.face_encodings = []
        
        # 初始化界面
        self.init_ui()
    
    def init_ui(self):
        """初始化对话框界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 表单区域
        form_layout = QFormLayout()
        
        # 姓名输入
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("请输入姓名")
        form_layout.addRow("姓名:", self.name_input)
        
        # 备注输入
        self.note_input = QLineEdit()
        self.note_input.setPlaceholderText("可选备注信息")
        form_layout.addRow("备注:", self.note_input)
        
        main_layout.addLayout(form_layout)
        
        # 预览区域
        preview_group = QGroupBox("人脸预览")
        preview_layout = QVBoxLayout()
        
        self.preview_label = QLabel("预览区域")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet("border: 1px solid #ccc;")
        preview_layout.addWidget(self.preview_label)
        
        # 人脸计数
        self.face_count_label = QLabel("已捕获人脸数量: 0")
        preview_layout.addWidget(self.face_count_label)
        
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)
        
        # 操作按钮区域
        buttons_layout = QHBoxLayout()
        
        self.capture_button = QPushButton("从摄像头捕获")
        self.capture_button.clicked.connect(self.capture_from_camera)
        buttons_layout.addWidget(self.capture_button)
        
        self.add_image_button = QPushButton("添加图片")
        self.add_image_button.clicked.connect(self.add_from_image)
        buttons_layout.addWidget(self.add_image_button)
        
        self.clear_button = QPushButton("清除所有")
        self.clear_button.clicked.connect(self.clear_all)
        buttons_layout.addWidget(self.clear_button)
        
        main_layout.addLayout(buttons_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # 注册按钮
        self.register_button = QPushButton("注册人脸")
        self.register_button.clicked.connect(self.register_face)
        self.register_button.setEnabled(False)
        main_layout.addWidget(self.register_button)
    
    def update_ui_state(self):
        """更新UI状态"""
        # 更新人脸计数
        self.face_count_label.setText(f"已捕获人脸数量: {len(self.face_encodings)}")
        
        # 更新进度条
        progress = min(100, len(self.face_encodings) * 20)  # 最多5张人脸，每张20%
        self.progress_bar.setValue(progress)
        
        # 更新注册按钮状态
        self.register_button.setEnabled(len(self.face_encodings) > 0 and self.name_input.text().strip())
    
    def show_face_preview(self, face_image):
        """显示人脸预览图像"""
        # 转换OpenCV图像为Qt图像
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 调整大小
        scaled_image = qt_image.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 显示图像
        self.preview_label.setPixmap(QPixmap.fromImage(scaled_image))
    
    def capture_from_camera(self):
        """从摄像头捕获人脸"""
        # 检查是否已达到最大人脸数量
        if len(self.face_encodings) >= 5:
            QMessageBox.warning(self, "警告", "最多只能添加5张人脸")
            return
        
        # 获取摄像头ID（从父窗口获取）
        camera_id = 0
        if self.parent() and hasattr(self.parent(), 'camera_combo'):
            camera_id = self.parent().camera_combo.currentData()
        
        # 创建临时摄像头捕获
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            return
        
        # 提示用户准备
        QMessageBox.information(self, "提示", "请面对摄像头，将显示3秒倒计时")
        
        # 倒计时
        for i in range(3, 0, -1):
            QApplication.processEvents()
            time.sleep(1)
        
        # 捕获多张图像
        captured_faces = 0
        max_attempts = 5
        
        for _ in range(max_attempts):
            # 读取图像
            ret, frame = cap.read()
            if not ret:
                continue
            
            try:
                # 检测人脸
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    # 提取人脸编码
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings:
                        # 保存人脸编码
                        self.face_encodings.append(face_encodings[0])
                        
                        # 显示预览
                        self.show_face_preview(frame)
                        
                        # 更新状态
                        self.update_ui_state()
                        
                        captured_faces += 1
                        
                        # 提示成功
                        QMessageBox.information(self, "成功", f"已捕获第 {captured_faces} 张人脸")
                        break
            except Exception as e:
                logger.error(f"捕获人脸时出错: {str(e)}")
        
        # 释放摄像头
        cap.release()
        
        if captured_faces == 0:
            QMessageBox.warning(self, "警告", "未能捕获到人脸，请重试")
    
    def add_from_image(self):
        """从图片添加人脸"""
        # 检查是否已达到最大人脸数量
        if len(self.face_encodings) >= 5:
            QMessageBox.warning(self, "警告", "最多只能添加5张人脸")
            return
        
        # 打开文件选择对话框
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择人脸图片", "", "图片文件 (*.jpg *.jpeg *.png)"
        )
        
        if not file_paths:
            return
        
        added_faces = 0
        
        # 处理每个选中的文件
        for file_path in file_paths:
            # 检查是否已达到最大数量
            if len(self.face_encodings) >= 5:
                break
            
            try:
                # 加载图像
                image = face_recognition.load_image_file(file_path)
                
                # 检测人脸
                face_locations = face_recognition.face_locations(image)
                if not face_locations:
                    QMessageBox.warning(self, "警告", f"图片 {file_path} 中未检测到人脸")
                    continue
                
                # 提取人脸编码
                face_encodings = face_recognition.face_encodings(image, face_locations)
                if face_encodings:
                    # 保存人脸编码
                    self.face_encodings.append(face_encodings[0])
                    
                    # 显示预览
                    # 将image转回BGR格式显示
                    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    self.show_face_preview(bgr_image)
                    
                    added_faces += 1
            except Exception as e:
                logger.error(f"处理图片 {file_path} 时出错: {str(e)}")
                QMessageBox.warning(self, "警告", f"处理图片失败: {str(e)}")
        
        # 更新状态
        self.update_ui_state()
        
        if added_faces > 0:
            QMessageBox.information(self, "成功", f"已成功添加 {added_faces} 张人脸")
    
    def clear_all(self):
        """清除所有捕获的人脸"""
        # 确认清除
        reply = QMessageBox.question(
            self, "确认清除",
            "确定要清除所有捕获的人脸吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 清空列表
            self.face_encodings.clear()
            
            # 清空预览
            self.preview_label.setText("预览区域")
            
            # 更新状态
            self.update_ui_state()
    
    def register_face(self):
        """注册人脸"""
        # 获取姓名
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入姓名")
            return
        
        # 检查人脸编码数量
        if not self.face_encodings:
            QMessageBox.warning(self, "警告", "请先添加人脸图像")
            return
        
        try:
            # 如果有多个人脸编码，计算平均值
            if len(self.face_encodings) > 1:
                avg_encoding = np.mean(self.face_encodings, axis=0)
            else:
                avg_encoding = self.face_encodings[0]
            
            # 添加到数据库
            face_db.add_face(name, avg_encoding)
            face_db.save_encodings()
            
            # 记录日志
            logger.info(f"高级注册成功: {name}, 使用了 {len(self.face_encodings)} 张人脸图像")
            
            # 显示成功消息
            QMessageBox.information(self, "成功", f"已成功注册 {name} 的人脸信息")
            
            # 接受对话框
            self.accept()
            
        except Exception as e:
            logger.error(f"注册人脸失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"注册失败: {str(e)}")
    
    def closeEvent(self, event):
        """
        窗口关闭事件处理
        确保正确停止线程和释放资源
        """
        # 停止识别系统
        self.stop_recognition()
        
        # 停止TTS引擎
        tts_engine.stop()
        
        # 更新日志
        logger.info("系统关闭")
        
        # 接受关闭事件
        event.accept()


def main():
    """
    主函数
    创建应用程序实例并启动主窗口
    """
    # 启用高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    
    # 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 记录启动信息
    logger.info("PyQt5图形界面启动")
    
    # 启动应用程序主循环
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
