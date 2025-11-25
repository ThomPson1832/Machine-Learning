import cv2
import numpy as np
import time
from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import threading
import logging
import csv
from datetime import datetime

# 设置日志记录
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('motion_analysis.log'),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# 导入自定义模块
from pose_detector import PoseDetector
from motion_analyzer import MotionAnalyzer
from motion_visualizer import MotionVisualizer

class DataProcessor(QtCore.QObject):
    """
    数据处理线程，负责姿态检测和角度计算
    """
    # 信号定义
    frame_processed = QtCore.pyqtSignal(object, object, object, object, object)  # 传递处理后的帧、姿态数据、3D关节点和加速度数据
    detection_failed = QtCore.pyqtSignal()  # 姿态检测失败信号
    
    def __init__(self, pose_detector, motion_analyzer):
        super().__init__()
        self.pose_detector = pose_detector
        self.motion_analyzer = motion_analyzer
        self.is_running = False
        self.start_time = time.time()  # 处理开始时间（初始化为当前时间）
        
    def start_processing(self):
        self.is_running = True
        self.start_time = time.time()
        logger.info("开始数据处理")
    
    def stop_processing(self):
        self.is_running = False
    
    def process_frame(self, frame):
        """
        处理单帧图像，进行姿态检测和角度计算
        """
        if not self.is_running:
            return
        
        try:
            # 检测姿态
            results, processed_frame = self.pose_detector.detect_pose(frame)
            
            if results and results.pose_landmarks:
                # 获取3D关节点坐标
                landmarks_3d = self.pose_detector.get_landmarks(results, mode='3d')
                
                if landmarks_3d:
                    # 将3D关节点添加到运动分析器的历史记录中
                    self.motion_analyzer.add_frame(landmarks_3d)
                    
                    # 计算关节角度
                    angles = {
                        'left_elbow': self.motion_analyzer.calculate_joint_angle('left_elbow'),
                        'right_elbow': self.motion_analyzer.calculate_joint_angle('right_elbow'),
                        'left_knee': self.motion_analyzer.calculate_joint_angle('left_knee'),
                        'right_knee': self.motion_analyzer.calculate_joint_angle('right_knee'),
                        'left_shoulder': self.motion_analyzer.calculate_joint_angle('left_shoulder'),
                        'right_shoulder': self.motion_analyzer.calculate_joint_angle('right_shoulder'),
                        'left_hip': self.motion_analyzer.calculate_joint_angle('left_hip'),
                        'right_hip': self.motion_analyzer.calculate_joint_angle('right_hip'),
                        'left_ankle': self.motion_analyzer.calculate_joint_angle('left_ankle'),
                        'right_ankle': self.motion_analyzer.calculate_joint_angle('right_ankle'),
                        'left_wrist': self.motion_analyzer.calculate_joint_angle('left_wrist'),
                        'right_wrist': self.motion_analyzer.calculate_joint_angle('right_wrist')
                    }
                    
                    # 计算关节速度
                    velocities = {
                        'left_elbow': self.motion_analyzer.calculate_velocity('left_elbow'),
                        'right_elbow': self.motion_analyzer.calculate_velocity('right_elbow'),
                        'left_knee': self.motion_analyzer.calculate_velocity('left_knee'),
                        'right_knee': self.motion_analyzer.calculate_velocity('right_knee'),
                        'left_shoulder': self.motion_analyzer.calculate_velocity('left_shoulder'),
                        'right_shoulder': self.motion_analyzer.calculate_velocity('right_shoulder'),
                        'left_hip': self.motion_analyzer.calculate_velocity('left_hip'),
                        'right_hip': self.motion_analyzer.calculate_velocity('right_hip'),
                        'left_ankle': self.motion_analyzer.calculate_velocity('left_ankle'),
                        'right_ankle': self.motion_analyzer.calculate_velocity('right_ankle'),
                        'left_wrist': self.motion_analyzer.calculate_velocity('left_wrist'),
                        'right_wrist': self.motion_analyzer.calculate_velocity('right_wrist')
                    }
                    
                    # 计算关节加速度
                    accelerations = {
                        'left_elbow': self.motion_analyzer.calculate_acceleration('left_elbow'),
                        'right_elbow': self.motion_analyzer.calculate_acceleration('right_elbow'),
                        'left_knee': self.motion_analyzer.calculate_acceleration('left_knee'),
                        'right_knee': self.motion_analyzer.calculate_acceleration('right_knee'),
                        'left_shoulder': self.motion_analyzer.calculate_acceleration('left_shoulder'),
                        'right_shoulder': self.motion_analyzer.calculate_acceleration('right_shoulder'),
                        'left_hip': self.motion_analyzer.calculate_acceleration('left_hip'),
                        'right_hip': self.motion_analyzer.calculate_acceleration('right_hip'),
                        'left_ankle': self.motion_analyzer.calculate_acceleration('left_ankle'),
                        'right_ankle': self.motion_analyzer.calculate_acceleration('right_ankle'),
                        'left_wrist': self.motion_analyzer.calculate_acceleration('left_wrist'),
                        'right_wrist': self.motion_analyzer.calculate_acceleration('right_wrist')
                    }
                    
                    # 发送处理后的帧、姿态数据和3D关节点
                    self.frame_processed.emit(processed_frame, angles, velocities, accelerations, landmarks_3d)
                    return
            
            # 没有检测到姿态，发送失败信号
            self.detection_failed.emit()
            
        except Exception as e:
            logger.error(f"数据处理错误: {str(e)}")
            self.detection_failed.emit()

class MotionAnalysisApp(QtWidgets.QMainWindow):
    """
    运动分析软件主程序
    """
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("基于AI的无标记3D运动分析系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中心Widget
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局
        self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        
        # 创建左侧视频显示区域
        self.video_frame = QtWidgets.QLabel()
        self.video_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.video_frame.setAlignment(QtCore.Qt.AlignCenter)
        self.video_frame.setStyleSheet("background-color: black;")
        self.main_layout.addWidget(self.video_frame, 5)
        
        # 创建右侧控制面板
        self.control_panel = QtWidgets.QWidget()
        self.control_panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.control_layout = QtWidgets.QVBoxLayout(self.control_panel)
        self.main_layout.addWidget(self.control_panel, 2)
        
        # 创建标题
        self.title_label = QtWidgets.QLabel("AI运动分析系统")
        self.title_label.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.control_layout.addWidget(self.title_label)
        self.control_layout.addSpacing(10)  # 添加间距
        
        # 相机控制分组
        camera_group = QtWidgets.QGroupBox("相机控制")
        camera_group.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        camera_layout = QtWidgets.QVBoxLayout()
        
        self.camera_label = QtWidgets.QLabel("选择摄像头:")
        camera_layout.addWidget(self.camera_label)
        
        self.camera_combo = QtWidgets.QComboBox()
        # 动态检测可用摄像头
        self.detect_cameras()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addSpacing(5)
        
        # 相机预览区域
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setFixedSize(320, 240)
        self.preview_label.setStyleSheet("border: 2px solid #ccc; border-radius: 5px;")
        self.preview_label.setText("点击预览按钮查看摄像头画面")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        camera_layout.addWidget(self.preview_label)
        
        # 预览控制按钮
        self.preview_button = QtWidgets.QPushButton("开始预览")
        self.preview_button.setStyleSheet("background-color: #2196F3; color: white; font-size: 12px; padding: 8px;")
        self.preview_button.clicked.connect(self.toggle_preview)
        camera_layout.addWidget(self.preview_button)
        
        camera_group.setLayout(camera_layout)
        self.control_layout.addWidget(camera_group)
        self.control_layout.addSpacing(10)
        
        # 分析控制分组
        analysis_group = QtWidgets.QGroupBox("分析控制")
        analysis_group.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        analysis_layout = QtWidgets.QVBoxLayout()
        
        # 开始/停止分析按钮
        self.start_button = QtWidgets.QPushButton("开始分析")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 12px; padding: 8px;")
        self.start_button.clicked.connect(self.start_analysis)
        analysis_layout.addWidget(self.start_button)
        
        self.stop_button = QtWidgets.QPushButton("停止分析")
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; font-size: 12px; padding: 8px;")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        analysis_layout.addWidget(self.stop_button)
        analysis_layout.addSpacing(5)
        
        # 3D可视化按钮
        self.visualize_button = QtWidgets.QPushButton("打开3D可视化")
        self.visualize_button.setStyleSheet("background-color: #9C27B0; color: white; font-size: 12px; padding: 8px;")
        self.visualize_button.clicked.connect(self.toggle_visualization)
        analysis_layout.addWidget(self.visualize_button)
        
        analysis_group.setLayout(analysis_layout)
        self.control_layout.addWidget(analysis_group)
        self.control_layout.addSpacing(10)
        
        # 视频控制分组
        video_group = QtWidgets.QGroupBox("视频操作")
        video_group.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        video_layout = QtWidgets.QVBoxLayout()
        
        # 视频录制按钮
        self.record_button = QtWidgets.QPushButton("开始录制")
        self.record_button.setStyleSheet("background-color: #FF5722; color: white; font-size: 12px; padding: 8px;")
        self.record_button.clicked.connect(self.toggle_recording)
        video_layout.addWidget(self.record_button)
        
        # 视频导入按钮
        self.load_video_button = QtWidgets.QPushButton("导入视频分析")
        self.load_video_button.setStyleSheet("background-color: #795548; color: white; font-size: 12px; padding: 8px;")
        self.load_video_button.clicked.connect(self.load_video)
        video_layout.addWidget(self.load_video_button)
        
        video_group.setLayout(video_layout)
        self.control_layout.addWidget(video_group)
        self.control_layout.addSpacing(10)
        
        # 数据管理分组
        data_group = QtWidgets.QGroupBox("数据管理")
        data_group.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        data_layout = QtWidgets.QVBoxLayout()
        
        # 数据导出按钮
        self.export_button = QtWidgets.QPushButton("导出数据")
        self.export_button.setStyleSheet("background-color: #FF9800; color: white; font-size: 12px; padding: 8px;")
        self.export_button.clicked.connect(self.export_data)
        data_layout.addWidget(self.export_button)
        
        # 数据清除按钮
        self.clear_button = QtWidgets.QPushButton("清除历史数据")
        self.clear_button.setStyleSheet("background-color: #f44336; color: white; font-size: 12px; padding: 8px;")
        self.clear_button.clicked.connect(self.clear_history)
        data_layout.addWidget(self.clear_button)
        
        data_group.setLayout(data_layout)
        self.control_layout.addWidget(data_group)
        self.control_layout.addSpacing(10)
        
        # 状态显示
        self.status_label = QtWidgets.QLabel("状态: 就绪")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("color: green; font-size: 14px; font-weight: bold;")
        self.status_label.setFixedHeight(30)
        self.control_layout.addWidget(self.status_label)
        
        # 运动参数显示分组
        param_group = QtWidgets.QGroupBox("运动参数")
        param_group.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        param_layout = QtWidgets.QVBoxLayout()
        
        # 创建滚动区域
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        
        # 创建关节角度显示
        self.joint_angle_label = QtWidgets.QLabel("关节角度:")
        self.joint_angle_label.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        scroll_layout.addWidget(self.joint_angle_label)
        
        self.angle_display = QtWidgets.QTextEdit()
        self.angle_display.setReadOnly(True)
        self.angle_display.setFixedHeight(90)
        self.angle_display.setStyleSheet("border: 1px solid #ccc; border-radius: 3px;")
        scroll_layout.addWidget(self.angle_display)
        
        scroll_layout.addSpacing(5)
        
        # 创建速度显示
        self.velocity_label = QtWidgets.QLabel("关节速度:")
        self.velocity_label.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        scroll_layout.addWidget(self.velocity_label)
        
        self.velocity_display = QtWidgets.QTextEdit()
        self.velocity_display.setReadOnly(True)
        self.velocity_display.setFixedHeight(90)
        self.velocity_display.setStyleSheet("border: 1px solid #ccc; border-radius: 3px;")
        scroll_layout.addWidget(self.velocity_display)
        
        scroll_layout.addSpacing(5)
        
        # 创建加速度显示
        self.acceleration_label = QtWidgets.QLabel("关节加速度:")
        self.acceleration_label.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        scroll_layout.addWidget(self.acceleration_label)
        
        self.acceleration_display = QtWidgets.QTextEdit()
        self.acceleration_display.setReadOnly(True)
        self.acceleration_display.setFixedHeight(90)
        self.acceleration_display.setStyleSheet("border: 1px solid #ccc; border-radius: 3px;")
        scroll_layout.addWidget(self.acceleration_display)
        
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setFixedHeight(320)
        param_layout.addWidget(scroll_area)
        
        param_group.setLayout(param_layout)
        self.control_layout.addWidget(param_group)
        
        # 添加垂直伸展空间
        self.control_layout.addStretch()
        
        # 添加垂直伸展空间
        self.control_layout.addStretch()
        
        # 初始化模块
        self.pose_detector = PoseDetector()
        self.motion_analyzer = MotionAnalyzer()
        self.visualizer = None
        
        # 初始化数据处理器
        self.data_processor = DataProcessor(self.pose_detector, self.motion_analyzer)
        self.data_processor.frame_processed.connect(self.on_frame_processed)
        self.data_processor.detection_failed.connect(self.on_detection_failed)
        
        # 创建数据处理线程
        self.processing_thread = QtCore.QThread()
        self.data_processor.moveToThread(self.processing_thread)
        self.processing_thread.start()
        
        # 初始化相机
        self.cap = None
        self.is_running = False
        self.is_visualizing = False
        self.is_previewing = False
        self.preview_cap = None
        
        # 创建定时器用于更新UI
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(30)  # 约33fps更新频率
        
        # 创建预览定时器
        self.preview_timer = QtCore.QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(30)
        
        # 存储历史数据
        self.joint_history = {
            'left_elbow': [],
            'right_elbow': [],
            'left_knee': [],
            'right_knee': []
        }
        
        # 视频录制相关变量
        self.is_recording = False
        self.recording_out = None
        self.recording_filename = None
    
    def on_camera_changed(self, index):
        """
        当选择的摄像头改变时停止当前预览
        """
        if self.is_previewing:
            self.toggle_preview()
    
    def toggle_preview(self):
        """
        切换相机预览状态
        """
        if self.is_previewing:
            # 停止预览
            self.stop_preview()
        else:
            # 开始预览
            self.start_preview()
    
    def start_preview(self):
        """
        开始相机预览
        """
        # 如果正在预览，先停止预览
        if self.is_previewing:
            self.stop_preview()
            
        # 获取选中的摄像头
        camera_index = int(self.camera_combo.currentText().split()[-1])
        
        # 打开相机
        self.preview_cap = cv2.VideoCapture(camera_index)
        
        if not self.preview_cap.isOpened():
            self.status_label.setText("状态: 相机预览失败")
            self.status_label.setStyleSheet("color: red;")
            logger.error(f"无法打开摄像头 {camera_index} 进行预览")
            return
        
        # 设置状态
        self.is_previewing = True
        self.preview_button.setText("停止预览")
        logger.info(f"开始预览摄像头 {camera_index}")
    
    def stop_preview(self):
        """
        停止相机预览
        """
        self.is_previewing = False
        
        # 关闭相机
        if self.preview_cap is not None:
            self.preview_cap.release()
            self.preview_cap = None
        
        # 重置预览标签
        self.preview_label.setText("点击预览按钮查看摄像头画面")
        self.preview_button.setText("开始预览")
        logger.info("停止相机预览")
    
    def update_preview(self):
        """
        更新相机预览画面
        """
        if not self.is_previewing or self.preview_cap is None:
            return
        
        ret, frame = self.preview_cap.read()
        if not ret:
            logger.error("相机预览读取失败")
            self.stop_preview()
            return
        
        # 镜像翻转帧（可选）
        frame = cv2.flip(frame, 1)
        
        # 调整帧大小以适应预览窗口
        frame = cv2.resize(frame, (320, 240))
        
        # 转换为Qt图像格式
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        
        # 显示图像
        self.preview_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
    
    def start_analysis(self, video_path=None):
        """
        开始运动分析
        
        参数:
            video_path: 可选，视频文件路径。如果提供，则分析视频文件；否则使用摄像头
        """
        if video_path:
            # 分析视频文件
            self.cap = cv2.VideoCapture(video_path)
            self.is_video_analysis = True
            self.video_path = video_path
            
            if not self.cap.isOpened():
                self.status_label.setText("状态: 视频文件打开失败")
                self.status_label.setStyleSheet("color: red;")
                logger.error(f"无法打开视频文件: {video_path}")
                return
            
            # 设置状态
            self.is_running = True
            self.status_label.setText("状态: 正在分析视频文件")
            self.status_label.setStyleSheet("color: blue;")
            logger.info(f"开始分析视频文件: {video_path}")
        else:
            # 分析摄像头
            camera_index = int(self.camera_combo.currentText().split()[-1])
            self.cap = cv2.VideoCapture(camera_index)
            self.is_video_analysis = False
            
            if not self.cap.isOpened():
                self.status_label.setText("状态: 相机打开失败")
                self.status_label.setStyleSheet("color: red;")
                logger.error(f"无法打开摄像头 {camera_index}")
                return
            
            # 设置状态
            self.is_running = True
            self.status_label.setText("状态: 正在分析")
            self.status_label.setStyleSheet("color: blue;")
            logger.info(f"开始使用摄像头 {camera_index} 进行运动分析")
        
        # 启用/禁用按钮
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # 启动数据处理
        self.data_processor.start_processing()
        
        # 启动相机/视频读取线程
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
    
    def stop_analysis(self):
        """
        停止运动分析
        """
        # 如果正在录制，先停止录制
        if self.is_recording:
            self.stop_recording()
        
        self.is_running = False
        self.data_processor.stop_processing()
        
        # 关闭相机/视频
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # 重置视频分析状态
        self.is_video_analysis = False
        self.video_path = None
        
        # 设置状态
        self.status_label.setText("状态: 就绪")
        self.status_label.setStyleSheet("color: green;")
        logger.info("停止运动分析")
        
        # 启用/禁用按钮
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # 清空显示
        self.video_frame.clear()
        self.angle_display.clear()
        self.velocity_display.clear()
        
        # 清空历史数据
        self.motion_analyzer.clear_history()
    
    def clear_history(self):
        """
        清除历史数据
        """
        try:
            if hasattr(self.data_processor, 'motion_analyzer'):
                self.data_processor.motion_analyzer.clear_history()
                QtWidgets.QMessageBox.information(self, "成功", "历史数据已清除")
            else:
                QtWidgets.QMessageBox.warning(self, "警告", "没有可清除的历史数据")
        except Exception as e:
            logger.error(f"清除历史数据失败: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "错误", f"清除历史数据失败: {str(e)}")
            
    def export_data(self):
        """
        导出运动分析数据
        """
        try:
            if not hasattr(self.data_processor, 'motion_analyzer'):
                QtWidgets.QMessageBox.warning(self, "警告", "没有可导出的历史数据")
                return
            
            # 打开文件选择对话框
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "导出数据", "motion_data.csv", "CSV Files (*.csv)"
            )
            
            if not filename:
                return  # 用户取消了导出
            
            # 导出数据
            success = self.data_processor.motion_analyzer.export_data(
                filename, 
                start_time=self.data_processor.start_time if hasattr(self.data_processor, 'start_time') else 0
            )
            
            if success:
                QtWidgets.QMessageBox.information(self, "成功", f"数据已成功导出到: {filename}")
            else:
                QtWidgets.QMessageBox.warning(self, "警告", "数据导出失败")
                
        except ImportError:
            # 如果缺少pandas，提示用户安装
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setText("导出功能需要pandas库")
            msg_box.setInformativeText("请运行: pip install pandas")
            msg_box.setWindowTitle("缺少依赖")
            msg_box.exec_()
        except Exception as e:
            logger.error(f"导出数据失败: {str(e)}")
            QtWidgets.QMessageBox.critical(self, "错误", f"导出数据失败: {str(e)}")

    def toggle_recording(self):
        """
        切换视频录制状态
        """
        if self.is_recording:
            # 停止录制
            self.stop_recording()
        else:
            # 开始录制
            self.start_recording()
    
    def start_recording(self):
        """
        开始视频录制
        """
        # 检查是否正在分析
        if not self.is_running:
            QtWidgets.QMessageBox.information(self, "录制视频", "请先开始分析才能录制视频")
            return
        
        # 获取保存文件路径
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存视频文件", "motion_recording_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".avi", "AVI Files (*.avi)"
        )
        
        if not file_path:
            return  # 用户取消了保存
        
        # 获取相机帧率和分辨率
        fps = 30.0  # 默认帧率
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 确保使用正确的分辨率（与分析时的一致）
        width, height = 640, 480
        
        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.recording_out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
        
        if not self.recording_out.isOpened():
            QtWidgets.QMessageBox.critical(self, "录制错误", "无法创建视频文件")
            logger.error(f"无法创建视频文件: {file_path}")
            return
        
        # 设置状态
        self.is_recording = True
        self.recording_filename = file_path
        self.record_button.setText("停止录制")
        self.record_button.setStyleSheet("background-color: red;")
        
        self.status_label.setText("状态: 正在分析和录制")
        self.status_label.setStyleSheet("color: orange;")
        
        logger.info(f"开始录制视频: {file_path}")
    
    def stop_recording(self):
        """
        停止视频录制
        """
        if self.recording_out is not None:
            self.recording_out.release()
            self.recording_out = None
        
        # 设置状态
        self.is_recording = False
        self.record_button.setText("开始录制")
        self.record_button.setStyleSheet("")
        
        self.status_label.setText("状态: 正在分析")
        self.status_label.setStyleSheet("color: orange;")
        
        logger.info(f"停止录制视频: {self.recording_filename}")
        self.recording_filename = None

    def toggle_visualization(self):
        """
        打开/关闭3D可视化
        """
        try:
            logger.info(f"toggle_visualization called, is_visualizing: {self.is_visualizing}")
            if not self.is_visualizing:
                logger.info("Creating MotionVisualizer instance...")
                self.visualizer = MotionVisualizer(self)
                self.is_visualizing = True
                self.visualize_button.setText("关闭3D可视化")
                logger.info("MotionVisualizer created successfully")
            else:
                logger.info("Closing MotionVisualizer...")
                self.visualizer.main_window.close()
                self.visualizer = None
                self.is_visualizing = False
                self.visualize_button.setText("打开3D可视化")
                logger.info("MotionVisualizer closed successfully")
        except Exception as e:
            logger.error(f"Error in toggle_visualization: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def load_video(self):
        """
        导入视频文件进行分析
        """
        # 获取视频文件路径
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        
        if not file_path:
            return  # 用户取消了选择
        
        # 开始分析视频文件
        self.start_analysis(video_path=file_path)
    
    def camera_loop(self):
        """
        相机/视频读取循环
        """
        frame_count = 0
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_analysis:
                    # 视频文件播放完毕
                    self.status_label.setText("状态: 视频分析完成")
                    self.status_label.setStyleSheet("color: green;")
                    logger.info(f"视频文件分析完成: {self.video_path}")
                    self.stop_analysis()
                else:
                    # 相机读取失败
                    self.status_label.setText("状态: 相机读取失败")
                    self.status_label.setStyleSheet("color: red;")
                    logger.error("相机读取失败")
                break
            
            # 镜像翻转帧（仅适用于摄像头）
            if not self.is_video_analysis:
                frame = cv2.flip(frame, 1)
            
            # 调整帧大小以适应窗口
            frame = cv2.resize(frame, (640, 480))
            
            # 如果正在录制且不是分析视频文件，则将帧写入视频文件
            if self.is_recording and self.recording_out is not None and not self.is_video_analysis:
                self.recording_out.write(frame)
            
            # 处理图像
            self.data_processor.process_frame(frame)
            
            # 计算FPS
            frame_count += 1
            if frame_count % 30 == 0:  # 每30帧计算一次FPS
                elapsed_time = time.time() - self.data_processor.start_time
                fps = frame_count / elapsed_time
                logger.info(f"当前FPS: {fps:.2f}")
                
                # 如果是视频分析，显示进度
                if self.is_video_analysis:
                    total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress = (frame_count / total_frames) * 100
                    self.status_label.setText(f"状态: 正在分析视频文件 ({progress:.1f}%)")
        
        self.is_running = False
    
    def on_frame_processed(self, processed_frame, angles, velocities, accelerations, landmarks_3d):
        """
        处理完成后的帧和姿态数据
        """
        try:
            # 转换为Qt图像格式
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            
            # 显示图像
            self.current_frame = QtGui.QPixmap.fromImage(qt_image)
            
            # 显示角度、速度和加速度
            self.update_angle_display(angles)
            self.update_velocity_display(velocities)
            self.update_acceleration_display(accelerations)
            
            # 更新3D可视化
            if self.is_visualizing and self.visualizer:
                logger.info(f"更新3D可视化，关节点数量: {len(landmarks_3d) if landmarks_3d else 0}")
                # 将3D关节点数据传递给可视化器
                self.visualizer.update_human_model(landmarks_3d)
                
                # 同时更新运动参数图表
                for joint, angle in angles.items():
                    if angle is not None:
                        logger.debug(f"添加角度数据: {joint} = {angle:.1f}°")
                        self.visualizer.add_angle_data(joint, angle, time.time() - self.data_processor.start_time)
                
                for joint, velocity in velocities.items():
                    if velocity is not None:
                        logger.debug(f"添加速度数据: {joint} = {velocity:.3f}")
                        self.visualizer.add_velocity_data(joint, velocity, time.time() - self.data_processor.start_time)
                
                for joint, acceleration in accelerations.items():
                    if acceleration is not None:
                        logger.debug(f"添加加速度数据: {joint} = {acceleration:.3f}")
                        self.visualizer.add_acceleration_data(joint, acceleration, time.time() - self.data_processor.start_time)
                
        except Exception as e:
            logger.error(f"帧处理错误: {str(e)}")
            
    def update_angle_display(self, angles):
        """
        更新角度显示
        """
        try:
            self.angle_text = ""
            for joint, angle in angles.items():
                if angle is not None:
                    self.angle_text += f"{joint}: {angle:.1f}°\n"
            self.angle_display.setText(self.angle_text)
        except Exception as e:
            logger.error(f"更新角度显示失败: {str(e)}")
    
    def update_velocity_display(self, velocities):
        """
        更新速度显示
        """
        try:
            self.velocity_text = ""
            for joint, velocity in velocities.items():
                if velocity is not None:
                    self.velocity_text += f"{joint}: {velocity:.3f}\n"
            self.velocity_display.setText(self.velocity_text)
        except Exception as e:
            logger.error(f"更新速度显示失败: {str(e)}")
            
    def update_acceleration_display(self, accelerations):
        """
        更新加速度显示
        """
        try:
            self.acceleration_text = ""
            for joint, acceleration in accelerations.items():
                if acceleration is not None:
                    self.acceleration_text += f"{joint}: {acceleration:.3f}\n"
            self.acceleration_display.setText(self.acceleration_text)
        except Exception as e:
            logger.error(f"更新加速度显示失败: {str(e)}")
    
    def on_detection_failed(self):
        """
        姿态检测失败时的处理
        """
        self.update_angle_display({})
        self.update_velocity_display({})
        self.update_acceleration_display({})
    
    def update_ui(self):
        """
        更新用户界面
        """
        # 更新视频帧
        if hasattr(self, 'current_frame'):
            self.video_frame.setPixmap(self.current_frame)
        
        # 更新角度显示
        if hasattr(self, 'angle_text'):
            self.angle_display.setText(self.angle_text)
        
        # 更新速度显示
        if hasattr(self, 'velocity_text'):
            self.velocity_display.setText(self.velocity_text)
        
        # 更新3D可视化界面
        if self.is_visualizing and self.visualizer:
            self.visualizer.update()
    
    def detect_cameras(self):
        """
        动态检测可用的摄像头设备
        """
        self.camera_combo.clear()
        logger.info("开始检测可用摄像头...")
        
        # 测试前10个摄像头索引
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"摄像头 {i}")
                cap.release()
                logger.info(f"检测到摄像头: {i}")
        
        # 如果没有检测到摄像头，添加默认选项
        if self.camera_combo.count() == 0:
            self.camera_combo.addItem("摄像头 0")
            logger.warning("未检测到任何摄像头，使用默认摄像头 0")
    
    def closeEvent(self, event):
        """
        窗口关闭事件
        """
        self.stop_analysis()
        if self.is_visualizing and self.visualizer:
            self.visualizer.main_window.close()
        
        # 停止预览
        if self.is_previewing:
            self.stop_preview()
            
        # 停止线程
        self.processing_thread.quit()
        self.processing_thread.wait()
        
        logger.info("程序已关闭")
        event.accept()

def main():
    """
    主函数
    """
    app = QtWidgets.QApplication(sys.argv)
    window = MotionAnalysisApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()