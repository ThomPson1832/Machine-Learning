import cv2
import numpy as np
import time
from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import threading

# 导入自定义模块
from pose_detector import PoseDetector
from motion_analyzer import MotionAnalyzer
from motion_visualizer import MotionVisualizer

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
        
        # 创建相机选择下拉框
        self.camera_label = QtWidgets.QLabel("选择摄像头:")
        self.control_layout.addWidget(self.camera_label)
        
        self.camera_combo = QtWidgets.QComboBox()
        # 动态检测可用摄像头
        self.detect_cameras()
        self.control_layout.addWidget(self.camera_combo)
        
        # 创建控制按钮
        self.start_button = QtWidgets.QPushButton("开始分析")
        self.start_button.clicked.connect(self.start_analysis)
        self.control_layout.addWidget(self.start_button)
        
        self.stop_button = QtWidgets.QPushButton("停止分析")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        self.control_layout.addWidget(self.stop_button)
        
        self.visualize_button = QtWidgets.QPushButton("打开3D可视化")
        self.visualize_button.clicked.connect(self.toggle_visualization)
        self.control_layout.addWidget(self.visualize_button)
        
        # 创建状态标签
        self.status_label = QtWidgets.QLabel("状态: 就绪")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("color: green;")
        self.control_layout.addWidget(self.status_label)
        
        # 创建关节角度显示
        self.joint_angle_label = QtWidgets.QLabel("关节角度:")
        self.control_layout.addWidget(self.joint_angle_label)
        
        self.angle_display = QtWidgets.QTextEdit()
        self.angle_display.setReadOnly(True)
        self.angle_display.setFixedHeight(100)
        self.control_layout.addWidget(self.angle_display)
        
        # 创建速度显示
        self.velocity_label = QtWidgets.QLabel("关节速度:")
        self.control_layout.addWidget(self.velocity_label)
        
        self.velocity_display = QtWidgets.QTextEdit()
        self.velocity_display.setReadOnly(True)
        self.velocity_display.setFixedHeight(100)
        self.control_layout.addWidget(self.velocity_display)
        
        # 添加垂直伸展空间
        self.control_layout.addStretch()
        
        # 初始化模块
        self.pose_detector = PoseDetector()
        self.motion_analyzer = MotionAnalyzer()
        self.visualizer = None
        
        # 初始化相机
        self.cap = None
        self.is_running = False
        self.is_visualizing = False
        
        # 创建定时器用于更新UI
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(30)  # 约33fps更新频率
        
        # 存储历史数据
        self.joint_history = {
            'left_elbow': [],
            'right_elbow': [],
            'left_knee': [],
            'right_knee': []
        }
    
    def start_analysis(self):
        """
        开始运动分析
        """
        # 获取选中的摄像头
        camera_index = int(self.camera_combo.currentText().split()[-1])
        
        # 打开相机
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            self.status_label.setText("状态: 相机打开失败")
            self.status_label.setStyleSheet("color: red;")
            return
        
        # 设置状态
        self.is_running = True
        self.status_label.setText("状态: 正在分析")
        self.status_label.setStyleSheet("color: blue;")
        
        # 启用/禁用按钮
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # 启动相机读取线程
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
    
    def stop_analysis(self):
        """
        停止运动分析
        """
        self.is_running = False
        
        # 关闭相机
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # 设置状态
        self.status_label.setText("状态: 就绪")
        self.status_label.setStyleSheet("color: green;")
        
        # 启用/禁用按钮
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # 清空显示
        self.video_frame.clear()
        self.angle_display.clear()
        self.velocity_display.clear()
    
    def toggle_visualization(self):
        """
        打开/关闭3D可视化
        """
        if not self.is_visualizing:
            self.visualizer = MotionVisualizer(self)
            self.is_visualizing = True
            self.visualize_button.setText("关闭3D可视化")
        else:
            self.visualizer.main_window.close()
            self.visualizer = None
            self.is_visualizing = False
            self.visualize_button.setText("打开3D可视化")
    
    def camera_loop(self):
        """
        相机读取循环
        """
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                # 相机读取失败，显示错误信息
                self.status_label.setText("状态: 相机读取失败")
                self.status_label.setStyleSheet("color: red;")
                break
            
            # 镜像翻转帧（可选）
            frame = cv2.flip(frame, 1)
            
            # 检测姿态
            results, frame = self.pose_detector.detect_pose(frame)
            
            # 调整帧大小以适应窗口
            frame = cv2.resize(frame, (640, 480))
            
            # 转换为Qt图像格式
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            
            # 显示图像
            self.current_frame = QtGui.QPixmap.fromImage(qt_image)
            
            # 处理姿态数据（如果检测到）
            if results and results.pose_landmarks:
                # 获取3D关节点坐标
                landmarks_3d = self.pose_detector.get_landmarks(results, mode='3d')
                
                if landmarks_3d:
                    # 计算关节角度
                    angles = {
                        'left_elbow': self.motion_analyzer.calculate_joint_angle(
                            landmarks_3d[11], landmarks_3d[13], landmarks_3d[15]
                        ),
                        'right_elbow': self.motion_analyzer.calculate_joint_angle(
                            landmarks_3d[12], landmarks_3d[14], landmarks_3d[16]
                        ),
                        'left_knee': self.motion_analyzer.calculate_joint_angle(
                            landmarks_3d[23], landmarks_3d[25], landmarks_3d[27]
                        ),
                        'right_knee': self.motion_analyzer.calculate_joint_angle(
                        landmarks_3d[24], landmarks_3d[26], landmarks_3d[28]
                    )
                }
                
                    # 更新历史数据
                    for joint, angle in angles.items():
                        self.joint_history[joint].append((time.time() - start_time, angle))
                        # 限制历史数据点数量
                        if len(self.joint_history[joint]) > 100:
                            self.joint_history[joint].pop(0)
                    
                    # 计算关节速度（使用前5帧的平均速度）
                    velocities = {
                        'left_elbow': self.motion_analyzer.calculate_velocity(
                            [x[1] for x in self.joint_history['left_elbow'][-5:]] if len(self.joint_history['left_elbow']) >= 5 else [0]
                        ),
                        'right_elbow': self.motion_analyzer.calculate_velocity(
                            [x[1] for x in self.joint_history['right_elbow'][-5:]] if len(self.joint_history['right_elbow']) >= 5 else [0]
                        ),
                        'left_knee': self.motion_analyzer.calculate_velocity(
                            [x[1] for x in self.joint_history['left_knee'][-5:]] if len(self.joint_history['left_knee']) >= 5 else [0]
                        ),
                        'right_knee': self.motion_analyzer.calculate_velocity(
                            [x[1] for x in self.joint_history['right_knee'][-5:]] if len(self.joint_history['right_knee']) >= 5 else [0]
                        )
                    }
                
                    # 更新3D可视化
                    if self.is_visualizing and self.visualizer:
                        self.visualizer.update_human_model(landmarks_3d)
                        
                        # 添加角度数据到图表
                        for joint, angle in angles.items():
                            self.visualizer.add_angle_data(joint, angle)
                        
                        # 添加速度数据到图表
                        for joint, velocity in velocities.items():
                            self.visualizer.add_velocity_data(joint, velocity)
                        
                        self.visualizer.update()
                    
                    # 显示角度和速度
                    self.update_angle_display(angles)
                    self.update_velocity_display(velocities)
            else:
                # 没有检测到姿态，清空角度和速度显示
                self.update_angle_display({})
                self.update_velocity_display({})
    
    def update_angle_display(self, angles):
        """
        更新关节角度显示
        """
        text = ""
        if angles:
            for joint, angle in angles.items():
                text += f"{joint}: {angle:.1f}°\n"
        else:
            text = "未检测到关节点"
        self.angle_text = text
    
    def update_velocity_display(self, velocities):
        """
        更新关节速度显示
        """
        text = ""
        if velocities:
            for joint, velocity in velocities.items():
                text += f"{joint}: {velocity:.1f}°/s\n"
        else:
            text = "未检测到关节点"
        self.velocity_text = text
    
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
    
    def detect_cameras(self):
        """
        动态检测可用的摄像头设备
        """
        self.camera_combo.clear()
        
        # 测试前10个摄像头索引
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"摄像头 {i}")
                cap.release()
        
        # 如果没有检测到摄像头，添加默认选项
        if self.camera_combo.count() == 0:
            self.camera_combo.addItem("摄像头 0")
    
    def closeEvent(self, event):
        """
        窗口关闭事件
        """
        self.stop_analysis()
        if self.is_visualizing and self.visualizer:
            self.visualizer.main_window.close()
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