import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import logging

# 设置日志记录
logger = logging.getLogger(__name__)

class MotionVisualizer:
    """
    运动数据可视化器，用于展示3D人体模型和运动参数图表
    """
    def __init__(self, parent=None):
        """
        初始化可视化器
        
        参数:
            parent: 父窗口实例
        """
        try:
            logger.info("正在初始化3D可视化器...")
            
            # 确保我们有一个QApplication实例
            self.app = QtWidgets.QApplication.instance()
            if self.app is None:
                logger.info("创建新的QApplication实例")
                self.app = QtWidgets.QApplication(sys.argv)
            else:
                logger.info("使用现有QApplication实例")
        
            self.parent = parent
            
            # 创建主窗口
            logger.info("创建主窗口...")
            self.main_window = QtWidgets.QMainWindow()
            self.main_window.setWindowTitle("3D运动分析可视化")
            self.main_window.setGeometry(100, 100, 1200, 800)
            
            # 创建中心 widget 和布局
            self.central_widget = QtWidgets.QWidget()
            self.main_window.setCentralWidget(self.central_widget)
            self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)
            
            # 创建左侧3D视图
            logger.info("创建3D视图...")
            self.view_3d = gl.GLViewWidget()
            self.view_3d.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.main_layout.addWidget(self.view_3d, 7)
            
            # 创建右侧控制面板
            self.control_panel = QtWidgets.QWidget()
            self.control_panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
            self.control_layout = QtWidgets.QVBoxLayout(self.control_panel)
            self.main_layout.addWidget(self.control_panel, 3)
            
            # 设置3D视图背景和坐标轴
            self.setup_3d_view()
            
            # 创建运动参数图表
            self.setup_charts()
            
            # 创建3D人体模型
            self.create_human_model()
            
            # 添加交互控制功能
            self.setup_interactive_controls()
            
            # 显示窗口
            logger.info("显示3D可视化窗口...")
            self.main_window.show()
            
        except Exception as e:
            logger.error(f"3D可视化器初始化失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def setup_3d_view(self):
        """
        设置3D视图的背景和坐标轴
        """
        # 设置黑色背景
        self.view_3d.setBackgroundColor('k')
        
        # 添加坐标轴
        self.grid = gl.GLGridItem()
        self.grid.setSize(x=20, y=20, z=20)
        self.grid.setSpacing(x=1, y=1, z=1)
        self.grid.setColor((0.3, 0.3, 0.3, 1))
        self.view_3d.addItem(self.grid)
        
        # 添加坐标系
        self.axis = gl.GLAxisItem()
        self.axis.setSize(x=5, y=5, z=5)
        self.view_3d.addItem(self.axis)
        
        # 设置相机位置
        self.view_3d.setCameraPosition(distance=10, elevation=30, azimuth=45)
        
        # 鼠标交互功能默认已支持，无需额外设置
        
        # 记录初始相机位置，用于重置
        self.initial_camera_pos = {
            'distance': 10,
            'elevation': 30,
            'azimuth': 45
        }
    
    def setup_charts(self):
        """
        创建运动参数图表
        """
        # 添加3D视图控制按钮
        self.control_layout.addWidget(QtWidgets.QLabel("3D视图控制"), alignment=QtCore.Qt.AlignCenter)
        
        # 视图控制按钮组
        view_controls = QtWidgets.QHBoxLayout()
        
        # 重置视图按钮
        self.reset_view_btn = QtWidgets.QPushButton("重置视图")
        self.reset_view_btn.clicked.connect(self.reset_view)
        view_controls.addWidget(self.reset_view_btn)
        
        # 前视图按钮
        self.front_view_btn = QtWidgets.QPushButton("前视图")
        self.front_view_btn.clicked.connect(lambda: self.set_view_preset('front'))
        view_controls.addWidget(self.front_view_btn)
        
        # 侧视图按钮
        self.side_view_btn = QtWidgets.QPushButton("侧视图")
        self.side_view_btn.clicked.connect(lambda: self.set_view_preset('side'))
        view_controls.addWidget(self.side_view_btn)
        
        # 顶视图按钮
        self.top_view_btn = QtWidgets.QPushButton("顶视图")
        self.top_view_btn.clicked.connect(lambda: self.set_view_preset('top'))
        view_controls.addWidget(self.top_view_btn)
        
        self.control_layout.addLayout(view_controls)
        
        # 创建角度图表
        self.angle_plot_widget = pg.PlotWidget(title="关节角度变化曲线")
        self.angle_plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.angle_plot_widget.showGrid(x=True, y=True)
        self.angle_plot_widget.setLabel('left', '角度 (度)')
        self.angle_plot_widget.setLabel('bottom', '时间 (帧)')
        self.control_layout.addWidget(self.angle_plot_widget)
        
        # 创建速度图表
        self.velocity_plot_widget = pg.PlotWidget(title="关节速度变化曲线")
        self.velocity_plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.velocity_plot_widget.showGrid(x=True, y=True)
        self.velocity_plot_widget.setLabel('left', '速度 (度/秒)')
        self.velocity_plot_widget.setLabel('bottom', '时间 (帧)')
        self.control_layout.addWidget(self.velocity_plot_widget)
        
        # 创建加速度图表
        self.acceleration_plot_widget = pg.PlotWidget(title="关节加速度变化曲线")
        self.acceleration_plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.acceleration_plot_widget.showGrid(x=True, y=True)
        self.acceleration_plot_widget.setLabel('left', '加速度 (度/秒²)')
        self.acceleration_plot_widget.setLabel('bottom', '时间 (帧)')
        self.control_layout.addWidget(self.acceleration_plot_widget)
        
        # 设置图表颜色和样式
        pg.setConfigOptions(antialias=True)
        
        # 存储图表曲线
        self.angle_curves = {}
        self.velocity_curves = {}
        self.acceleration_curves = {}
        
        # 添加图例
        self.angle_plot_widget.addLegend()
        self.velocity_plot_widget.addLegend()
        self.acceleration_plot_widget.addLegend()
    
    def create_human_model(self):
        """
        创建3D人体模型
        """
        # 定义不同身体部位的颜色
        self.joint_colors = {
            # 头部
            'head': (1.0, 0.8, 0.2, 1),  # 金色
            # 躯干
            'torso': (0.0, 0.8, 1.0, 1),  # 天蓝色
            # 左手臂
            'left_arm': (1.0, 0.0, 0.0, 1),  # 红色
            # 右手臂
            'right_arm': (0.0, 1.0, 0.0, 1),  # 绿色
            # 左腿
            'left_leg': (0.0, 0.0, 1.0, 1),  # 蓝色
            # 右腿
            'right_leg': (1.0, 0.0, 1.0, 1)   # 紫色
        }
        
        # 为每个关节点分配颜色
        joint_colors_list = np.zeros((33, 4))
        
        # 头部（0-10号关节点）
        joint_colors_list[0:11] = self.joint_colors['head']
        
        # 躯干（11-24号关节点中的躯干部分）
        joint_colors_list[11] = self.joint_colors['torso']  # 左肩
        joint_colors_list[12] = self.joint_colors['torso']  # 右肩
        joint_colors_list[23] = self.joint_colors['torso']  # 左髋
        joint_colors_list[24] = self.joint_colors['torso']  # 右髋
        
        # 左手臂（13-22号关节点中的左臂部分）
        joint_colors_list[13] = self.joint_colors['left_arm']  # 左肘
        joint_colors_list[15] = self.joint_colors['left_arm']  # 左腕
        joint_colors_list[17] = self.joint_colors['left_arm']  # 左小拇指
        joint_colors_list[19] = self.joint_colors['left_arm']  # 左食指
        joint_colors_list[21] = self.joint_colors['left_arm']  # 左拇指
        
        # 右手臂
        joint_colors_list[14] = self.joint_colors['right_arm']  # 右肘
        joint_colors_list[16] = self.joint_colors['right_arm']  # 右腕
        joint_colors_list[18] = self.joint_colors['right_arm']  # 右小拇指
        joint_colors_list[20] = self.joint_colors['right_arm']  # 右食指
        joint_colors_list[22] = self.joint_colors['right_arm']  # 右拇指
        
        # 左腿
        joint_colors_list[25] = self.joint_colors['left_leg']  # 左膝
        joint_colors_list[27] = self.joint_colors['left_leg']  # 左踝
        joint_colors_list[29] = self.joint_colors['left_leg']  # 左足跟
        joint_colors_list[31] = self.joint_colors['left_leg']  # 左脚趾
        
        # 右腿
        joint_colors_list[26] = self.joint_colors['right_leg']  # 右膝
        joint_colors_list[28] = self.joint_colors['right_leg']  # 右踝
        joint_colors_list[30] = self.joint_colors['right_leg']  # 右足跟
        joint_colors_list[32] = self.joint_colors['right_leg']  # 右脚趾
        
        # 创建关节点
        self.joint_points = gl.GLScatterPlotItem(
            pos=np.zeros((33, 3)),  # 33个关节点
            size=0.12,
            color=joint_colors_list,
            pxMode=True
        )
        self.view_3d.addItem(self.joint_points)
        
        # 创建骨骼连接
        self.bone_lines = gl.GLLinePlotItem(
            pos=np.zeros((0, 3)),  # 初始无连接
            color=(0.8, 0.8, 0.8, 1),  # 默认灰色
            width=2.5,
            mode='lines'
        )
        self.view_3d.addItem(self.bone_lines)
        
        # 定义骨骼连接及其颜色
        self.bone_connections = [
            # 头部
            (0, 1, 'head'), (1, 2, 'head'), (2, 3, 'head'), (3, 7, 'head'),  # 左眼
            (0, 4, 'head'), (4, 5, 'head'), (5, 6, 'head'), (6, 8, 'head'),  # 右眼
            (0, 9, 'head'), (0, 10, 'head'),  # 嘴巴
            # 躯干
            (11, 12, 'torso'),  # 肩部连接
            (11, 23, 'torso'), (12, 24, 'torso'), (23, 24, 'torso'),  # 躯干
            # 左手臂
            (11, 13, 'left_arm'), (13, 15, 'left_arm'),  # 左上臂和左前臂
            (15, 17, 'left_arm'), (17, 19, 'left_arm'), (19, 21, 'left_arm'),  # 左手
            (15, 21, 'left_arm'), (15, 19, 'left_arm'), (15, 17, 'left_arm'),  # 左手连接
            # 右手臂
            (12, 14, 'right_arm'), (14, 16, 'right_arm'),  # 右上臂和右前臂
            (16, 18, 'right_arm'), (18, 20, 'right_arm'), (20, 22, 'right_arm'),  # 右手
            (16, 22, 'right_arm'), (16, 20, 'right_arm'), (16, 18, 'right_arm'),  # 右手连接
            # 左腿
            (23, 25, 'left_leg'), (25, 27, 'left_leg'), (27, 29, 'left_leg'), (29, 31, 'left_leg'),  # 左腿
            (27, 31, 'left_leg'),  # 左脚
            # 右腿
            (24, 26, 'right_leg'), (26, 28, 'right_leg'), (28, 30, 'right_leg'), (30, 32, 'right_leg'),  # 右腿
            (28, 32, 'right_leg')   # 右脚
        ]
    
    def update_human_model(self, landmarks):
        """
        更新3D人体模型的关节点和骨骼位置
        
        参数:
            landmarks: 关节点坐标列表
        """
        if not landmarks:
            return
        
        try:
            # 转换为numpy数组
            positions = np.array(landmarks)
            
            # 优化坐标系转换
            # 1. 平移到中心位置（以骨盆为中心）
            if len(positions) > 23:  # 确保有骨盆关节点
                center_point = (positions[23] + positions[24]) / 2  # 左右髋部的中点
                positions = positions - center_point
            
            # 2. 缩放坐标，使人体模型大小适中
            scale_factor = 5.0
            positions *= scale_factor
            
            # 3. 调整坐标方向
            positions[:, 1] *= -1  # 翻转y轴，使向上为正
            positions[:, 2] *= -1  # 翻转z轴，使深度方向更直观
            
            # 4. 确保模型位于可视范围内
            min_pos = np.min(positions, axis=0)
            max_pos = np.max(positions, axis=0)
            model_size = np.max(max_pos - min_pos)
            
            # 动态调整缩放比例，确保模型大小适中
            if model_size > 10:  # 模型过大
                scale_factor *= 0.8
                positions *= 0.8
            elif model_size < 5:  # 模型过小
                scale_factor *= 1.2
                positions *= 1.2
            
            # 更新关节点位置
            self.joint_points.setData(pos=positions)
            
            # 更新骨骼连接
            bone_positions = []
            bone_colors = []
            valid_connections = 0
            
            for connection in self.bone_connections:
                try:
                    start_joint = positions[connection[0]]
                    end_joint = positions[connection[1]]
                    bone_positions.append(start_joint)
                    bone_positions.append(end_joint)
                    
                    # 获取连接对应的颜色
                    body_part = connection[2] if len(connection) > 2 else 'torso'
                    color = self.joint_colors[body_part]
                    # 为每一段骨骼添加颜色
                    bone_colors.append(color)
                    bone_colors.append(color)
                    
                    valid_connections += 1
                except IndexError:
                    continue
            
            if bone_positions:
                bone_positions_np = np.array(bone_positions)
                bone_colors_np = np.array(bone_colors)
                self.bone_lines.setData(pos=bone_positions_np, color=bone_colors_np)
            
        except Exception as e:
            import logging
            logging.error(f"更新3D模型失败: {str(e)}")
    
    def get_color_for_joint(self, joint_name):
        """
        根据关节名称获取颜色
        
        参数:
            joint_name: 关节名称
            
        返回:
            color: 颜色元组 (r, g, b)
        """
        color_map = {
            'left_elbow': (1, 0, 0),      # 红色
            'right_elbow': (0, 1, 0),     # 绿色
            'left_knee': (0, 0, 1),       # 蓝色
            'right_knee': (1, 1, 0),      # 黄色
            'left_shoulder': (1, 0, 1),   # 紫色
            'right_shoulder': (0, 1, 1),  # 青色
            'left_hip': (1, 0.5, 0),      # 橙色
            'right_hip': (0, 0.5, 0.5),   # 靛蓝色
            'left_ankle': (0.5, 0.5, 0),  # 橄榄色
            'right_ankle': (0.5, 0, 0.5), # 深紫色
            'left_wrist': (0, 0.75, 0.75),# 蓝绿色
            'right_wrist': (0.75, 0, 0.75)# 洋红色
        }
        
        return color_map.get(joint_name, (0.5, 0.5, 0.5))
    
    def add_angle_data(self, joint_name, angle, time=None):
        """
        添加关节角度数据到图表
        
        参数:
            joint_name: 关节名称
            angle: 角度值
            time: 时间戳
        """
        if joint_name not in self.angle_curves:
            # 创建新曲线
            curve = self.angle_plot_widget.plot(
                name=joint_name,
                pen=pg.mkPen(width=2, color=self.get_color_for_joint(joint_name))
            )
            self.angle_curves[joint_name] = {
                'curve': curve,
                'data': []
            }
        
        # 添加新数据点
        self.angle_curves[joint_name]['data'].append(angle)
        
        # 更新曲线
        self.angle_curves[joint_name]['curve'].setData(
            self.angle_curves[joint_name]['data']
        )
        
        # 限制数据点数量
        max_points = getattr(self, 'max_data_points', 100)
        if len(self.angle_curves[joint_name]['data']) > max_points:
            self.angle_curves[joint_name]['data'] = self.angle_curves[joint_name]['data'][-max_points:]
    
    def add_velocity_data(self, joint_name, velocity, time=None):
        """
        添加关节速度数据到图表
        
        参数:
            joint_name: 关节名称
            velocity: 速度值
            time: 时间戳
        """
        if joint_name not in self.velocity_curves:
            # 创建新曲线
            curve = self.velocity_plot_widget.plot(
                name=joint_name,
                pen=pg.mkPen(width=2, color=self.get_color_for_joint(joint_name))
            )
            self.velocity_curves[joint_name] = {
                'curve': curve,
                'data': []
            }
        
        # 添加新数据点
        self.velocity_curves[joint_name]['data'].append(velocity)
        
        # 更新曲线
        self.velocity_curves[joint_name]['curve'].setData(
            self.velocity_curves[joint_name]['data']
        )
        
        # 限制数据点数量
        max_points = getattr(self, 'max_data_points', 100)
        if len(self.velocity_curves[joint_name]['data']) > max_points:
            self.velocity_curves[joint_name]['data'] = self.velocity_curves[joint_name]['data'][-max_points:]
    
    def add_acceleration_data(self, joint_name, acceleration, time=None):
        """
        添加关节加速度数据到图表
        
        参数:
            joint_name: 关节名称
            acceleration: 加速度值
            time: 时间戳
        """
        if joint_name not in self.acceleration_curves:
            # 创建新曲线
            curve = self.acceleration_plot_widget.plot(
                name=joint_name,
                pen=pg.mkPen(width=2, color=self.get_color_for_joint(joint_name))
            )
            self.acceleration_curves[joint_name] = {
                'curve': curve,
                'data': []
            }
        
        # 添加新数据点
        self.acceleration_curves[joint_name]['data'].append(acceleration)
        
        # 更新曲线
        self.acceleration_curves[joint_name]['curve'].setData(
            self.acceleration_curves[joint_name]['data']
        )
        
        # 限制数据点数量
        max_points = getattr(self, 'max_data_points', 100)
        if len(self.acceleration_curves[joint_name]['data']) > max_points:
            self.acceleration_curves[joint_name]['data'] = self.acceleration_curves[joint_name]['data'][-max_points:]
    
    def setup_interactive_controls(self):
        """
        设置交互控制功能
        """
        # 视角控制组
        view_group = QtWidgets.QGroupBox("视角控制")
        view_layout = QtWidgets.QGridLayout()
        view_group.setLayout(view_layout)
        
        # 视角方向按钮
        reset_button = QtWidgets.QPushButton("重置视角")
        reset_button.clicked.connect(self.reset_view)
        view_layout.addWidget(reset_button, 0, 0, 1, 3)
        
        front_button = QtWidgets.QPushButton("前视图")
        front_button.clicked.connect(lambda: self.set_camera_view(0, 0, 10))
        view_layout.addWidget(front_button, 1, 1)
        
        left_button = QtWidgets.QPushButton("左视图")
        left_button.clicked.connect(lambda: self.set_camera_view(-90, 0, 10))
        view_layout.addWidget(left_button, 2, 0)
        
        right_button = QtWidgets.QPushButton("右视图")
        right_button.clicked.connect(lambda: self.set_camera_view(90, 0, 10))
        view_layout.addWidget(right_button, 2, 2)
        
        top_button = QtWidgets.QPushButton("顶视图")
        top_button.clicked.connect(lambda: self.set_camera_view(0, 90, 10))
        view_layout.addWidget(top_button, 2, 1)
        
        bottom_button = QtWidgets.QPushButton("底视图")
        bottom_button.clicked.connect(lambda: self.set_camera_view(0, -90, 10))
        view_layout.addWidget(bottom_button, 3, 1)
        
        back_button = QtWidgets.QPushButton("后视图")
        back_button.clicked.connect(lambda: self.set_camera_view(180, 0, 10))
        view_layout.addWidget(back_button, 4, 1)
        
        self.control_layout.addWidget(view_group)
        
        # 模型控制组
        model_group = QtWidgets.QGroupBox("模型控制")
        model_layout = QtWidgets.QVBoxLayout()
        model_group.setLayout(model_layout)
        
        # 缩放控制
        scale_label = QtWidgets.QLabel("模型缩放:")
        model_layout.addWidget(scale_label)
        
        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setRange(5, 20)
        self.scale_slider.setValue(10)
        self.scale_slider.valueChanged.connect(self.update_scale)
        model_layout.addWidget(self.scale_slider)
        
        # 自动旋转选项
        self.auto_rotate_checkbox = QtWidgets.QCheckBox("自动旋转模型")
        self.auto_rotate_checkbox.stateChanged.connect(self.toggle_auto_rotate)
        model_layout.addWidget(self.auto_rotate_checkbox)
        
        self.rotation_angle = 0
        self.rotation_timer = QtCore.QTimer()
        self.rotation_timer.timeout.connect(self.rotate_model)
        
        self.control_layout.addWidget(model_group)
        
        # 显示选项组
        display_group = QtWidgets.QGroupBox("显示选项")
        display_layout = QtWidgets.QVBoxLayout()
        display_group.setLayout(display_layout)
        
        # 关节点显示选项
        self.show_joints_checkbox = QtWidgets.QCheckBox("显示关节点")
        self.show_joints_checkbox.setChecked(True)
        self.show_joints_checkbox.stateChanged.connect(self.toggle_joints)
        display_layout.addWidget(self.show_joints_checkbox)
        
        # 骨骼显示选项
        self.show_bones_checkbox = QtWidgets.QCheckBox("显示骨骼")
        self.show_bones_checkbox.setChecked(True)
        self.show_bones_checkbox.stateChanged.connect(self.toggle_bones)
        display_layout.addWidget(self.show_bones_checkbox)
        
        # 坐标轴显示选项
        self.show_axis_checkbox = QtWidgets.QCheckBox("显示坐标轴")
        self.show_axis_checkbox.setChecked(True)
        self.show_axis_checkbox.stateChanged.connect(self.toggle_axis)
        display_layout.addWidget(self.show_axis_checkbox)
        
        self.control_layout.addWidget(display_group)
        
        # 数据显示选项组
        data_group = QtWidgets.QGroupBox("数据显示")
        data_layout = QtWidgets.QVBoxLayout()
        data_group.setLayout(data_layout)
        
        # 最大数据点设置
        max_points_label = QtWidgets.QLabel("图表最大数据点:")
        data_layout.addWidget(max_points_label)
        
        self.max_points_spinbox = QtWidgets.QSpinBox()
        self.max_points_spinbox.setRange(50, 500)
        self.max_points_spinbox.setValue(100)
        self.max_points_spinbox.valueChanged.connect(self.set_max_data_points)
        data_layout.addWidget(self.max_points_spinbox)
        
        self.control_layout.addWidget(data_group)
        
        # 添加垂直伸展空间
        self.control_layout.addStretch()
    
    def reset_view(self):
        """
        重置相机视角
        """
        self.view_3d.setCameraPosition(distance=10, elevation=30, azimuth=45)
        logger.info("重置相机视角")
    
    def set_camera_view(self, azimuth, elevation, distance):
        """
        设置相机视角
        
        参数:
            azimuth: 方位角
            elevation: 仰角
            distance: 距离
        """
        self.view_3d.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)
        logger.info(f"设置相机视角 - 方位角: {azimuth}, 仰角: {elevation}, 距离: {distance}")
    
    def update_scale(self, value):
        """
        更新模型缩放
        
        参数:
            value: 缩放值
        """
        scale_factor = value / 10.0
        self.joint_points.setSize(0.1 * scale_factor)
        logger.info(f"更新模型缩放: {scale_factor}")
    
    def toggle_auto_rotate(self, state):
        """
        切换自动旋转功能
        
        参数:
            state: 复选框状态
        """
        if state == QtCore.Qt.Checked:
            self.rotation_timer.start(30)  # 约33fps
            logger.info("启用模型自动旋转")
        else:
            self.rotation_timer.stop()
            logger.info("禁用模型自动旋转")
    
    def rotate_model(self):
        """
        旋转模型
        """
        self.rotation_angle += 1
        if self.rotation_angle >= 360:
            self.rotation_angle = 0
        
        self.view_3d.setCameraPosition(azimuth=self.rotation_angle, elevation=30, distance=10)
    
    def toggle_joints(self, state):
        """
        切换关节点显示
        
        参数:
            state: 复选框状态
        """
        if state == QtCore.Qt.Checked:
            self.joint_points.show()
            logger.info("显示关节点")
        else:
            self.joint_points.hide()
            logger.info("隐藏关节点")
    
    def toggle_bones(self, state):
        """
        切换骨骼显示
        
        参数:
            state: 复选框状态
        """
        if state == QtCore.Qt.Checked:
            self.bone_lines.show()
            logger.info("显示骨骼")
        else:
            self.bone_lines.hide()
            logger.info("隐藏骨骼")
    
    def toggle_axis(self, state):
        """
        切换坐标轴显示
        
        参数:
            state: 复选框状态
        """
        if hasattr(self, 'axis'):
            if state == QtCore.Qt.Checked:
                self.axis.show()
                logger.info("显示坐标轴")
            else:
                self.axis.hide()
                logger.info("隐藏坐标轴")
        
        if hasattr(self, 'grid'):
            if state == QtCore.Qt.Checked:
                self.grid.show()
                logger.info("显示网格")
            else:
                self.grid.hide()
                logger.info("隐藏网格")
    
    def set_max_data_points(self, value):
        """
        设置最大数据点数量
        
        参数:
            value: 最大数据点数量
        """
        self.max_data_points = value
        logger.info(f"设置最大数据点数量: {value}")
        
        # 更新所有曲线
        for joint_name in self.angle_curves:
            if len(self.angle_curves[joint_name]['data']) > value:
                self.angle_curves[joint_name]['data'] = self.angle_curves[joint_name]['data'][-value:]
                self.angle_curves[joint_name]['curve'].setData(self.angle_curves[joint_name]['data'])
        
        for joint_name in self.velocity_curves:
            if len(self.velocity_curves[joint_name]['data']) > value:
                self.velocity_curves[joint_name]['data'] = self.velocity_curves[joint_name]['data'][-value:]
                self.velocity_curves[joint_name]['curve'].setData(self.velocity_curves[joint_name]['data'])
        
        for joint_name in self.acceleration_curves:
            if len(self.acceleration_curves[joint_name]['data']) > value:
                self.acceleration_curves[joint_name]['data'] = self.acceleration_curves[joint_name]['data'][-value:]
                self.acceleration_curves[joint_name]['curve'].setData(self.acceleration_curves[joint_name]['data'])
    
    def update(self):
        """
        更新可视化界面
        """
        self.app.processEvents()
    
    def reset_view(self):
        """
        重置3D视图到初始位置
        """
        self.view_3d.setCameraPosition(
            distance=self.initial_camera_pos['distance'],
            elevation=self.initial_camera_pos['elevation'],
            azimuth=self.initial_camera_pos['azimuth']
        )
    
    def set_view_preset(self, preset_name):
        """
        设置预定义的视图角度
        
        参数:
            preset_name: 视图名称 ('front', 'side', 'top')
        """
        if preset_name == 'front':
            self.view_3d.setCameraPosition(distance=10, elevation=90, azimuth=0)
        elif preset_name == 'side':
            self.view_3d.setCameraPosition(distance=10, elevation=90, azimuth=90)
        elif preset_name == 'top':
            self.view_3d.setCameraPosition(distance=10, elevation=0, azimuth=0)
    
    def run(self):
        """
        运行可视化器
        """
        if sys.flags.interactive != 1:
            self.app.exec_()

if __name__ == "__main__":
    # 测试可视化器
    visualizer = MotionVisualizer()
    visualizer.run()