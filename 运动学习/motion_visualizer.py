import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import logging

# 设置日志记录
logger = logging.getLogger(__name__)

class MotionVisualizer(QtWidgets.QMainWindow):
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
            
            # 确保QApplication实例存在
            self.app = QtWidgets.QApplication.instance()
            if self.app is None:
                logger.info("创建新的QApplication实例")
                self.app = QtWidgets.QApplication(sys.argv)
            else:
                logger.info("使用现有QApplication实例")
            
            # 调用父类构造函数
            super(MotionVisualizer, self).__init__(parent)
            
            # 设置窗口属性
            self.setWindowTitle("3D运动分析可视化")
            self.setGeometry(100, 100, 1200, 800)
            
            # 创建中心widget和布局
            self.central_widget = QtWidgets.QWidget()
            self.setCentralWidget(self.central_widget)
            self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)
            
            # 创建3D视图
            logger.info("创建3D视图...")
            self.view_3d = gl.GLViewWidget()
            self.view_3d.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            
            # 将3D视图添加到主布局
            self.main_layout.addWidget(self.view_3d, 7)
            
            # 创建右侧控制面板
            self.control_panel = QtWidgets.QWidget()
            self.control_panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
            self.control_layout = QtWidgets.QVBoxLayout(self.control_panel)
            self.main_layout.addWidget(self.control_panel, 3)
            
            # 设置3D视图背景和坐标轴
            self.setup_3d_view()
            
            # 创建3D人体模型
            self.create_human_model()
            
            # 添加交互控制功能
            self.setup_interactive_controls()
            
            # 教学标注相关变量初始化
            self.annotations = []  # 存储所有标注对象
            self.current_color = (1.0, 0.0, 0.0, 1.0)  # 默认红色
            
            # 添加更新计数器
            self.frame_count = 0
            
            # 设置图表
            self.setup_charts()
            
            # 显示窗口
            self.show()
            
            logger.info("3D可视化器初始化完成...")
            
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
        
        # 背面视图按钮
        self.back_view_btn = QtWidgets.QPushButton("背面视图")
        self.back_view_btn.clicked.connect(lambda: self.set_view_preset('back'))
        view_controls.addWidget(self.back_view_btn)
        
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
        
        # 添加波浪图组件
        self.wave_plot_widget = pg.PlotWidget(title="关节数据波浪图")
        self.wave_plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.wave_plot_widget.showGrid(x=True, y=True)
        self.wave_plot_widget.setLabel('left', '数值')
        self.wave_plot_widget.setLabel('bottom', '时间 (帧)')
        self.control_layout.addWidget(self.wave_plot_widget)
        
        # 设置图表颜色和样式
        pg.setConfigOptions(antialias=True)
        
        # 存储图表曲线
        self.angle_curves = {}
        self.velocity_curves = {}
        self.acceleration_curves = {}
        self.wave_curves = {}  # 波浪图曲线
        
        # 添加图例
        self.angle_plot_widget.addLegend()
        self.velocity_plot_widget.addLegend()
        self.acceleration_plot_widget.addLegend()
        self.wave_plot_widget.addLegend()
        
        # 添加表格组件
        self.data_table = QtWidgets.QTableWidget()
        self.data_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # 设置表格列名
        self.data_table.setColumnCount(4)
        self.data_table.setHorizontalHeaderLabels(['关节', '角度 (度)', '速度 (度/秒)', '加速度 (度/秒²)'])
        # 设置表格行数（根据关节数量）
        self.joint_names = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee',
                          'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                          'left_ankle', 'right_ankle', 'left_wrist', 'right_wrist']
        self.data_table.setRowCount(len(self.joint_names))
        # 设置行名
        for row, joint_name in enumerate(self.joint_names):
            self.data_table.setItem(row, 0, QtWidgets.QTableWidgetItem(joint_name))
        # 调整表格大小
        self.data_table.resizeColumnsToContents()
        self.control_layout.addWidget(self.data_table)
    
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
            size=0.15,  # 增加关节点大小
            color=joint_colors_list,
            pxMode=True
        )
        self.view_3d.addItem(self.joint_points)
        
        # 创建骨骼连接
        self.bone_lines = gl.GLLinePlotItem(
            pos=np.zeros((0, 3)),  # 初始无连接
            color=(0.8, 0.8, 0.8, 1),  # 默认灰色
            width=3.0,  # 增加骨骼粗细
            mode='lines'
        )
        self.view_3d.addItem(self.bone_lines)
        
        # 创建关节点标签
        self.joint_labels = []
        joint_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
            'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
            'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
        ]
        
        # 定义骨骼连接
        self.bone_connections = [
            # 头部连接
            (0, 1, 'head'), (1, 2, 'head'), (2, 3, 'head'),  # 左眼周围
            (0, 4, 'head'), (4, 5, 'head'), (5, 6, 'head'),  # 右眼周围
            (0, 7, 'head'), (7, 8, 'head'),  # 耳朵
            (9, 10, 'head'),  # 嘴巴
            
            # 躯干连接
            (11, 23, 'torso'), (12, 24, 'torso'),  # 肩膀到髋部
            
            # 左手臂连接
            (11, 13, 'left_arm'), (13, 15, 'left_arm'),  # 左肩膀-左肘-左腕
            (15, 17, 'left_arm'), (15, 19, 'left_arm'), (15, 21, 'left_arm'),  # 左腕到手指
            
            # 右手臂连接
            (12, 14, 'right_arm'), (14, 16, 'right_arm'),  # 右肩膀-右肘-右腕
            (16, 18, 'right_arm'), (16, 20, 'right_arm'), (16, 22, 'right_arm'),  # 右腕到手指
            
            # 左腿连接
            (23, 25, 'left_leg'), (25, 27, 'left_leg'), (27, 29, 'left_leg'), (29, 31, 'left_leg'),  # 左腿
            (27, 31, 'left_leg'),  # 左脚
            
            # 右腿连接
            (24, 26, 'right_leg'), (26, 28, 'right_leg'), (28, 30, 'right_leg'), (30, 32, 'right_leg'),  # 右腿
            (28, 32, 'right_leg')   # 右脚
        ]
        
        # 创建光照效果
        self.setup_lighting()
        
    def setup_lighting(self):
        """
        设置3D场景的光照效果
        """
        try:
            # 尝试添加环境光
            self.ambient_light = gl.GLLight(
                pos=(0, 0, 10),  # 光源位置
                color=(0.8, 0.8, 0.8, 1),  # 环境光颜色
                intensity=0.5,  # 环境光强度
                lightType='ambient'
            )
            self.view_3d.addItem(self.ambient_light)
            
            # 添加方向光
            self.directional_light = gl.GLLight(
                pos=(10, 10, 10),  # 光源位置
                color=(1, 1, 1, 1),  # 方向光颜色
                intensity=1.0,  # 方向光强度
                lightType='directional'
            )
            self.view_3d.addItem(self.directional_light)
            
            # 添加点光源
            self.point_light = gl.GLLight(
                pos=(-10, 10, 10),  # 光源位置
                color=(1, 1, 1, 1),  # 点光源颜色
                intensity=1.0,  # 点光源强度
                lightType='point'
            )
            self.view_3d.addItem(self.point_light)
        except AttributeError:
            # 如果GLLight不存在，跳过光照设置
            logger.warning("GLLight属性不存在，跳过光照设置")
            pass
        
        # 设置材质属性
        try:
            self.view_3d.setMaterial(gl.GLMaterial(
                diffuse=(0.5, 0.5, 0.5, 1),
                specular=(1, 1, 1, 1),
                shininess=100
            ))
        except AttributeError:
            # 如果setMaterial方法不存在，跳过材质设置
            logger.warning("setMaterial方法不存在，跳过材质设置")
            pass
    
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
        
        # 录制功能：如果正在录制，保存当前帧的关键点数据
        if hasattr(self, 'is_recording') and self.is_recording:
            if hasattr(self, 'recorded_frames'):
                self.recorded_frames.append(np.array(landmarks))
        
        try:
            # 转换为numpy数组，并确保只使用前3个坐标值（x, y, z）
            positions = np.array(landmarks)
            if positions.shape[1] > 3:
                positions = positions[:, :3]  # 只取前3个坐标值，忽略置信度
            
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
            
            # 仅在需要时更新模型大小（减少计算）
            if not hasattr(self, 'last_model_size') or self.frame_count % 30 == 0:
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
                
                self.last_model_size = model_size
            
            # 更新关节点位置
            self.joint_points.setData(pos=positions)
            
            # 更新骨骼连接
            if hasattr(self, 'bone_connections'):
                # 预先分配足够大的数组以避免多次内存分配
                max_bones = len(self.bone_connections)
                bone_positions = np.empty((max_bones * 2, 3), dtype=np.float32)
                bone_colors = np.empty((max_bones * 2, 3), dtype=np.float32)  # 使用3个分量（RGB）
                valid_connections = 0
                
                for connection in self.bone_connections:
                    try:
                        start_joint = positions[connection[0]]
                        end_joint = positions[connection[1]]
                        
                        bone_positions[valid_connections * 2] = start_joint
                        bone_positions[valid_connections * 2 + 1] = end_joint
                        
                        # 获取连接对应的颜色，并转换为3个分量（RGB）
                        body_part = connection[2] if len(connection) > 2 else 'torso'
                        color = self.joint_colors[body_part]
                        # 为每一段骨骼添加颜色（只使用RGB分量）
                        bone_colors[valid_connections * 2] = color[:3]
                        bone_colors[valid_connections * 2 + 1] = color[:3]
                        
                        valid_connections += 1
                    except IndexError:
                        continue
                
                if valid_connections > 0:
                    # 只使用有效的连接部分
                    bone_positions = bone_positions[:valid_connections * 2]
                    bone_colors = bone_colors[:valid_connections * 2]
                    self.bone_lines.setData(pos=bone_positions, color=bone_colors)
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
                'times': np.array([], dtype=np.float64),
                'values': np.array([], dtype=np.float64),
                'update_count': 0
            }
        
        # 添加新数据点
        if time is None:
            time = len(self.angle_curves[joint_name]['times'])  # 使用索引作为时间
            
        # 追加数据点
        times = self.angle_curves[joint_name]['times']
        values = self.angle_curves[joint_name]['values']
        
        # 使用numpy数组更高效地追加数据
        times = np.append(times, time)
        values = np.append(values, angle)
        
        self.angle_curves[joint_name]['times'] = times
        self.angle_curves[joint_name]['values'] = values
        
        # 限制更新频率
        self.angle_curves[joint_name]['update_count'] += 1
        
        # 每5个数据点更新一次曲线，减少渲染次数
        if self.angle_curves[joint_name]['update_count'] % 5 == 0:
            # 限制数据点数量
            max_points = getattr(self, 'max_data_points', 100)
            if len(times) > max_points:
                self.angle_curves[joint_name]['times'] = times[-max_points:]
                self.angle_curves[joint_name]['values'] = values[-max_points:]
            
            # 更新曲线
            self.angle_curves[joint_name]['curve'].setData(
                self.angle_curves[joint_name]['times'],
                self.angle_curves[joint_name]['values']
            )
    
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
                'times': np.array([], dtype=np.float64),
                'values': np.array([], dtype=np.float64),
                'update_count': 0
            }
        
        # 添加新数据点
        if time is None:
            time = len(self.velocity_curves[joint_name]['times'])  # 使用索引作为时间
            
        # 追加数据点
        times = self.velocity_curves[joint_name]['times']
        values = self.velocity_curves[joint_name]['values']
        
        # 使用numpy数组更高效地追加数据
        times = np.append(times, time)
        values = np.append(values, velocity)
        
        self.velocity_curves[joint_name]['times'] = times
        self.velocity_curves[joint_name]['values'] = values
        
        # 限制更新频率
        self.velocity_curves[joint_name]['update_count'] += 1
        
        # 每5个数据点更新一次曲线，减少渲染次数
        if self.velocity_curves[joint_name]['update_count'] % 5 == 0:
            # 限制数据点数量
            max_points = getattr(self, 'max_data_points', 100)
            if len(times) > max_points:
                self.velocity_curves[joint_name]['times'] = times[-max_points:]
                self.velocity_curves[joint_name]['values'] = values[-max_points:]
            
            # 更新曲线
            self.velocity_curves[joint_name]['curve'].setData(
                self.velocity_curves[joint_name]['times'],
                self.velocity_curves[joint_name]['values']
            )
    
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
                'times': np.array([], dtype=np.float64),
                'values': np.array([], dtype=np.float64),
                'update_count': 0
            }
        
        # 添加新数据点
        if time is None:
            time = len(self.acceleration_curves[joint_name]['times'])  # 使用索引作为时间
            
        # 追加数据点
        times = self.acceleration_curves[joint_name]['times']
        values = self.acceleration_curves[joint_name]['values']
        
        # 使用numpy数组更高效地追加数据
        times = np.append(times, time)
        values = np.append(values, acceleration)
        
        self.acceleration_curves[joint_name]['times'] = times
        self.acceleration_curves[joint_name]['values'] = values
        
        # 限制更新频率
        self.acceleration_curves[joint_name]['update_count'] += 1
        
        # 每5个数据点更新一次曲线，减少渲染次数
        if self.acceleration_curves[joint_name]['update_count'] % 5 == 0:
            # 限制数据点数量
            max_points = getattr(self, 'max_data_points', 100)
            if len(times) > max_points:
                self.acceleration_curves[joint_name]['times'] = times[-max_points:]
                self.acceleration_curves[joint_name]['values'] = values[-max_points:]
            
            # 更新曲线
            self.acceleration_curves[joint_name]['curve'].setData(
                self.acceleration_curves[joint_name]['times'],
                self.acceleration_curves[joint_name]['values']
            )
    
    def update_data_table(self, angles, velocities, accelerations):
        """
        更新数据表格，显示各关节的实时数值
        
        参数:
            angles: 各关节角度字典
            velocities: 各关节速度字典
            accelerations: 各关节加速度字典
        """
        # 限制更新频率，每5帧更新一次表格
        if self.frame_count % 5 != 0:
            return
        
        for joint_name in self.joint_names:
            if joint_name in self.joint_names:
                row = self.joint_names.index(joint_name)
                
                # 更新角度
                if joint_name in angles and angles[joint_name] is not None:
                    angle_value = f"{angles[joint_name]:.2f}"
                    current_text = self.data_table.item(row, 1).text() if self.data_table.item(row, 1) else ""
                    if current_text != angle_value:
                        self.data_table.setItem(row, 1, QtWidgets.QTableWidgetItem(angle_value))
                else:
                    self.data_table.setItem(row, 1, QtWidgets.QTableWidgetItem("-"))
                
                # 更新速度
                if joint_name in velocities and velocities[joint_name] is not None:
                    velocity_value = f"{velocities[joint_name]:.3f}"
                    current_text = self.data_table.item(row, 2).text() if self.data_table.item(row, 2) else ""
                    if current_text != velocity_value:
                        self.data_table.setItem(row, 2, QtWidgets.QTableWidgetItem(velocity_value))
                else:
                    self.data_table.setItem(row, 2, QtWidgets.QTableWidgetItem("-"))
                
                # 更新加速度
                if joint_name in accelerations and accelerations[joint_name] is not None:
                    acceleration_value = f"{accelerations[joint_name]:.3f}"
                    current_text = self.data_table.item(row, 3).text() if self.data_table.item(row, 3) else ""
                    if current_text != acceleration_value:
                        self.data_table.setItem(row, 3, QtWidgets.QTableWidgetItem(acceleration_value))
                else:
                    self.data_table.setItem(row, 3, QtWidgets.QTableWidgetItem("-"))
    
    def add_wave_data(self, joint_name, value, data_type='angle'):
        """
        添加关节数据到波浪图
        
        参数:
            joint_name: 关节名称
            value: 数值
            data_type: 数据类型 ('angle', 'velocity', 'acceleration')
        """
        # 创建波浪图曲线
        if joint_name not in self.wave_curves:
            self.wave_curves[joint_name] = {
                'angle': None,
                'velocity': None,
                'acceleration': None
            }
        
        if self.wave_curves[joint_name][data_type] is None:
            # 创建新曲线
            curve = self.wave_plot_widget.plot(
                name=f"{joint_name}_{data_type}",
                pen=pg.mkPen(width=2, color=self.get_color_for_joint(joint_name))
            )
            self.wave_curves[joint_name][data_type] = {
                'curve': curve,
                'data': np.array([], dtype=np.float64),
                'update_count': 0
            }
        
        # 添加新数据点
        data = self.wave_curves[joint_name][data_type]['data']
        data = np.append(data, value)
        self.wave_curves[joint_name][data_type]['data'] = data
        
        # 限制更新频率，每5个数据点更新一次曲线
        self.wave_curves[joint_name][data_type]['update_count'] += 1
        if self.wave_curves[joint_name][data_type]['update_count'] % 5 == 0:
            # 限制数据点数量
            max_points = getattr(self, 'max_data_points', 200)
            if len(data) > max_points:
                self.wave_curves[joint_name][data_type]['data'] = data[-max_points:]
            
            # 更新曲线
            self.wave_curves[joint_name][data_type]['curve'].setData(
                self.wave_curves[joint_name][data_type]['data']
            )
        
        # 限制数据点数量
        max_points = getattr(self, 'max_data_points', 100)
        if len(self.wave_curves[joint_name][data_type]['data']) > max_points:
            self.wave_curves[joint_name][data_type]['data'] = self.wave_curves[joint_name][data_type]['data'][-max_points:]
    
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
        display_layout = QtWidgets.QGridLayout()
        display_group.setLayout(display_layout)
        
        # 关节点显示选项
        self.show_joints_checkbox = QtWidgets.QCheckBox("显示关节点")
        self.show_joints_checkbox.setChecked(True)
        self.show_joints_checkbox.stateChanged.connect(self.toggle_joints)
        display_layout.addWidget(self.show_joints_checkbox, 0, 0)
        
        # 骨骼显示选项
        self.show_bones_checkbox = QtWidgets.QCheckBox("显示骨骼")
        self.show_bones_checkbox.setChecked(True)
        self.show_bones_checkbox.stateChanged.connect(self.toggle_bones)
        display_layout.addWidget(self.show_bones_checkbox, 0, 1)
        
        # 坐标轴显示选项
        self.show_axis_checkbox = QtWidgets.QCheckBox("显示坐标轴")
        self.show_axis_checkbox.setChecked(True)
        self.show_axis_checkbox.stateChanged.connect(self.toggle_axis)
        display_layout.addWidget(self.show_axis_checkbox, 1, 0)
        
        # 网格显示选项
        self.show_grid_checkbox = QtWidgets.QCheckBox("显示网格")
        self.show_grid_checkbox.setChecked(True)
        self.show_grid_checkbox.stateChanged.connect(self.toggle_grid)
        display_layout.addWidget(self.show_grid_checkbox, 1, 1)
        
        self.control_layout.addWidget(display_group)
        
        # 数据显示选项组
        data_group = QtWidgets.QGroupBox("数据显示")
        data_layout = QtWidgets.QVBoxLayout()
        data_group.setLayout(data_layout)
        
        # 最大数据点设置
        max_points_layout = QtWidgets.QHBoxLayout()
        max_points_label = QtWidgets.QLabel("图表最大数据点:")
        max_points_layout.addWidget(max_points_label)
        
        self.max_points_spinbox = QtWidgets.QSpinBox()
        self.max_points_spinbox.setRange(50, 1000)
        self.max_points_spinbox.setValue(200)
        self.max_points_spinbox.valueChanged.connect(self.set_max_data_points)
        max_points_layout.addWidget(self.max_points_spinbox)
        data_layout.addLayout(max_points_layout)
        
        # 关节角度显示选项
        data_layout.addWidget(QtWidgets.QLabel("显示关节角度:"))
        
        joint_angles_layout = QtWidgets.QGridLayout()
        
        # 主要关节复选框
        joints_to_display = [
            ('left_shoulder', '左肩'),
            ('right_shoulder', '右肩'),
            ('left_elbow', '左肘'),
            ('right_elbow', '右肘'),
            ('left_hip', '左髋'),
            ('right_hip', '右髋'),
            ('left_knee', '左膝'),
            ('right_knee', '右膝'),
            ('left_ankle', '左脚踝'),
            ('right_ankle', '右脚踝'),
            ('left_wrist', '左手腕'),
            ('right_wrist', '右手腕')
        ]
        
        self.joint_angle_checkboxes = {}
        
        for i, (joint_id, joint_name) in enumerate(joints_to_display):
            checkbox = QtWidgets.QCheckBox(joint_name)
            checkbox.setChecked(True)
            checkbox.joint_id = joint_id
            self.joint_angle_checkboxes[joint_id] = checkbox
            row = i // 2
            col = i % 2
            joint_angles_layout.addWidget(checkbox, row, col)
        
        data_layout.addLayout(joint_angles_layout)
        
        # 数据显示更新频率
        update_rate_layout = QtWidgets.QHBoxLayout()
        update_rate_label = QtWidgets.QLabel("更新频率:")
        update_rate_layout.addWidget(update_rate_label)
        
        self.update_rate_spinbox = QtWidgets.QSpinBox()
        self.update_rate_spinbox.setRange(1, 60)
        self.update_rate_spinbox.setValue(10)
        self.update_rate_spinbox.valueChanged.connect(self.set_update_rate)
        update_rate_layout.addWidget(self.update_rate_spinbox)
        update_rate_layout.addWidget(QtWidgets.QLabel("fps"))
        data_layout.addLayout(update_rate_layout)
        
        self.control_layout.addWidget(data_group)
        
        # 录制控制组
        recording_group = QtWidgets.QGroupBox("动作录制")
        recording_layout = QtWidgets.QHBoxLayout()
        recording_group.setLayout(recording_layout)
        
        self.record_button = QtWidgets.QPushButton("开始录制")
        self.record_button.clicked.connect(self.toggle_recording)
        recording_layout.addWidget(self.record_button)
        
        self.play_button = QtWidgets.QPushButton("播放")
        self.play_button.clicked.connect(self.play_recording)
        self.play_button.setEnabled(False)
        recording_layout.addWidget(self.play_button)
        
        self.save_button = QtWidgets.QPushButton("保存")
        self.save_button.clicked.connect(self.save_recording)
        self.save_button.setEnabled(False)
        recording_layout.addWidget(self.save_button)
        
        self.load_button = QtWidgets.QPushButton("加载")
        self.load_button.clicked.connect(self.load_recording)
        recording_layout.addWidget(self.load_button)
        
        # 播放速度控制
        speed_layout = QtWidgets.QHBoxLayout()
        speed_label = QtWidgets.QLabel("播放速度:")
        speed_layout.addWidget(speed_label)
        
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setRange(1, 10)  # 0.1x 到 1.0x
        self.speed_slider.setValue(10)  # 默认1.0x
        self.speed_slider.setTickInterval(1)
        self.speed_slider.setSingleStep(1)
        self.speed_slider.valueChanged.connect(self._update_playback_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QtWidgets.QLabel("1.0x")
        speed_layout.addWidget(self.speed_label)
        
        recording_layout.addLayout(speed_layout)
        
        # 关键帧控制
        keyframe_layout = QtWidgets.QHBoxLayout()
        self.keyframe_button = QtWidgets.QPushButton("标记关键帧")
        self.keyframe_button.clicked.connect(self.mark_keyframe)
        keyframe_layout.addWidget(self.keyframe_button)
        
        self.keyframe_list = QtWidgets.QListWidget()
        self.keyframe_list.itemDoubleClicked.connect(self._jump_to_keyframe)
        keyframe_layout.addWidget(self.keyframe_list)
        
        recording_layout.addLayout(keyframe_layout)
        
        self.control_layout.addWidget(recording_group)
        
        # 动作比对控制组
        comparison_group = QtWidgets.QGroupBox("动作比对")
        comparison_layout = QtWidgets.QHBoxLayout()
        comparison_group.setLayout(comparison_layout)
        
        self.import_standard_button = QtWidgets.QPushButton("导入标准动作")
        self.import_standard_button.clicked.connect(self.load_standard_action)
        comparison_layout.addWidget(self.import_standard_button)
        
        self.compare_button = QtWidgets.QPushButton("开始比对")
        self.compare_button.clicked.connect(self.start_comparison)
        self.compare_button.setEnabled(False)  # 默认禁用，直到导入标准动作
        comparison_layout.addWidget(self.compare_button)
        
        self.control_layout.addWidget(comparison_group)
        
        # 关节活动范围统计组
        range_group = QtWidgets.QGroupBox("关节活动范围统计")
        range_layout = QtWidgets.QVBoxLayout()
        range_group.setLayout(range_layout)
        
        # 创建统计结果显示区域
        self.range_table = QtWidgets.QTableWidget(0, 4)
        self.range_table.setHorizontalHeaderLabels(["关节名称", "最小角度(°)", "最大角度(°)", "活动范围(°)"])
        self.range_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        range_layout.addWidget(self.range_table)
        
        # 添加计算按钮
        calculate_layout = QtWidgets.QHBoxLayout()
        self.calculate_range_button = QtWidgets.QPushButton("计算活动范围")
        self.calculate_range_button.clicked.connect(self.calculate_joint_ranges)
        calculate_layout.addWidget(self.calculate_range_button)
        
        self.clear_range_button = QtWidgets.QPushButton("清除数据")
        self.clear_range_button.clicked.connect(self.clear_joint_ranges)
        calculate_layout.addWidget(self.clear_range_button)
        
        range_layout.addLayout(calculate_layout)
        
        self.control_layout.addWidget(range_group)
        
        # 教学标注工具组
        annotation_group = QtWidgets.QGroupBox("教学标注工具")
        annotation_layout = QtWidgets.QVBoxLayout()
        annotation_group.setLayout(annotation_layout)
        
        # 标注类型选择
        type_layout = QtWidgets.QHBoxLayout()
        type_label = QtWidgets.QLabel("标注类型:")
        type_layout.addWidget(type_label)
        
        self.annotation_type_combo = QtWidgets.QComboBox()
        self.annotation_type_combo.addItems(["线条", "箭头", "文字"])
        type_layout.addWidget(self.annotation_type_combo)
        
        annotation_layout.addLayout(type_layout)
        
        # 标注颜色选择
        color_layout = QtWidgets.QHBoxLayout()
        color_label = QtWidgets.QLabel("颜色:")
        color_layout.addWidget(color_label)
        
        self.color_button = QtWidgets.QPushButton("")
        self.color_button.setFixedSize(30, 30)
        self.color_button.setStyleSheet("background-color: red;")
        self.color_button.clicked.connect(self.select_color)
        self.current_color = (1.0, 0.0, 0.0, 1.0)  # 默认红色
        color_layout.addWidget(self.color_button)
        
        annotation_layout.addLayout(color_layout)
        
        # 线条宽度设置
        width_layout = QtWidgets.QHBoxLayout()
        width_label = QtWidgets.QLabel("线条宽度:")
        width_layout.addWidget(width_label)
        
        self.width_spinbox = QtWidgets.QDoubleSpinBox()
        self.width_spinbox.setRange(0.1, 5.0)
        self.width_spinbox.setValue(1.0)
        self.width_spinbox.setSingleStep(0.1)
        width_layout.addWidget(self.width_spinbox)
        
        annotation_layout.addLayout(width_layout)
        
        # 文字大小设置
        font_layout = QtWidgets.QHBoxLayout()
        font_label = QtWidgets.QLabel("文字大小:")
        font_layout.addWidget(font_label)
        
        self.font_size_spinbox = QtWidgets.QSpinBox()
        self.font_size_spinbox.setRange(8, 36)
        self.font_size_spinbox.setValue(14)
        font_layout.addWidget(self.font_size_spinbox)
        
        annotation_layout.addLayout(font_layout)
        
        # 文字内容输入
        text_layout = QtWidgets.QHBoxLayout()
        text_label = QtWidgets.QLabel("文字:")
        text_layout.addWidget(text_label)
        
        self.text_input = QtWidgets.QLineEdit("标注文字")
        text_layout.addWidget(self.text_input)
        
        annotation_layout.addLayout(text_layout)
        
        # 操作按钮
        action_layout = QtWidgets.QHBoxLayout()
        self.add_annotation_button = QtWidgets.QPushButton("添加标注")
        self.add_annotation_button.clicked.connect(self.add_annotation)
        action_layout.addWidget(self.add_annotation_button)
        
        self.clear_annotations_button = QtWidgets.QPushButton("清除所有标注")
        self.clear_annotations_button.clicked.connect(self.clear_all_annotations)
        action_layout.addWidget(self.clear_annotations_button)
        
        annotation_layout.addLayout(action_layout)
        
        self.control_layout.addWidget(annotation_group)
        
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
    
    def toggle_grid(self, state):
        """
        切换网格显示
        
        参数:
            state: 复选框状态
        """
        if hasattr(self, 'grid'):
            if state == QtCore.Qt.Checked:
                self.grid.show()
                logger.info("显示网格")
            else:
                self.grid.hide()
                logger.info("隐藏网格")
    
    def set_max_data_points(self, value):
        """
        设置图表最大数据点数量
        
        参数:
            value: 最大数据点数量
        """
        self.max_data_points = value
        logger.info(f"设置图表最大数据点: {value}")
        
        # 更新所有曲线的数据点限制
        for curves_dict in [self.angle_curves, self.velocity_curves, self.acceleration_curves]:
            for joint_name, curve_data in curves_dict.items():
                # 只保留最新的value个数据点
                if len(curve_data['times']) > value:
                    curve_data['times'] = curve_data['times'][-value:]
                    curve_data['values'] = curve_data['values'][-value:]
                    # 更新曲线显示
                    curve_data['curve'].setData(
                        curve_data['times'],
                        curve_data['values']
                    )
    
    def set_update_rate(self, value):
        """
        设置数据更新频率
        
        参数:
            value: 更新频率（fps）
        """
        self.update_rate = value
        # 计算更新间隔（毫秒）
        update_interval = int(1000 / value)
        
        # 更新定时器间隔
        if hasattr(self, 'update_timer'):
            self.update_timer.setInterval(update_interval)
            logger.info(f"设置更新频率: {value} fps (间隔: {update_interval}ms)")
    
    def toggle_recording(self):
        """
        开始/停止动作录制
        """
        if not hasattr(self, 'is_recording'):
            self.is_recording = False
            self.recorded_frames = []
            self.keyframes = []
        
        self.is_recording = not self.is_recording
        
        if self.is_recording:
            self.record_button.setText("停止录制")
            self.play_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.keyframe_button.setEnabled(True)
            self.recorded_frames = []  # 清空之前的录制
            self.keyframes = []  # 清空关键帧
            self.keyframe_list.clear()  # 清空关键帧列表
            logger.info("开始动作录制...")
        else:
            self.record_button.setText("开始录制")
            self.play_button.setEnabled(len(self.recorded_frames) > 0)
            self.save_button.setEnabled(len(self.recorded_frames) > 0)
            self.keyframe_button.setEnabled(True)
            logger.info(f"停止动作录制，共录制 {len(self.recorded_frames)} 帧")
    
    def play_recording(self):
        """
        播放/暂停录制的动作
        """
        if not hasattr(self, 'recorded_frames') or not self.recorded_frames:
            logger.warning("没有可播放的录制动作")
            return
        
        # 初始化播放状态
        if not hasattr(self, 'is_playing'):
            self.is_playing = False
            self.current_frame_index = 0
            self.playback_timer = QtCore.QTimer()
            self.playback_timer.timeout.connect(self._play_next_frame)
            self.playback_speed = 1.0  # 默认播放速度
        
        if not self.is_playing:
            # 开始播放
            self.is_playing = True
            self.play_button.setText("暂停播放")
            self.record_button.setEnabled(False)
            self.load_button.setEnabled(False)
            
            # 设置播放间隔（默认30fps）
            frame_interval = int(1000 / (30 * self.playback_speed))
            self.playback_timer.setInterval(frame_interval)
            self.playback_timer.start()
            
            logger.info(f"开始播放录制的动作，共 {len(self.recorded_frames)} 帧")
        else:
            # 暂停播放
            self.is_playing = False
            self.play_button.setText("继续播放")
            self.record_button.setEnabled(True)
            self.load_button.setEnabled(True)
            self.playback_timer.stop()
            
            logger.info(f"暂停播放，当前播放到第 {self.current_frame_index} 帧")
            
    def _update_playback_speed(self, value):
        """
        更新播放速度
        
        参数:
            value: 滑块值 (1-10)
        """
        # 计算实际播放速度 (0.1x 到 1.0x)
        self.playback_speed = value / 10.0
        self.speed_label.setText(f"{self.playback_speed:.1f}x")
        
        # 如果正在播放，更新定时器间隔
        if hasattr(self, 'is_playing') and self.is_playing:
            frame_interval = int(1000 / (30 * self.playback_speed))
            self.playback_timer.setInterval(frame_interval)
            
        logger.info(f"设置播放速度: {self.playback_speed:.1f}x")
    
    def _play_next_frame(self):
        """
        播放下一帧动作数据
        """
        if self.current_frame_index < len(self.recorded_frames):
            # 更新人体模型
            frame_data = self.recorded_frames[self.current_frame_index]
            self.update_human_model(frame_data)
            
            # 更新当前帧索引
            self.current_frame_index += 1
        else:
            # 播放结束
            self.is_playing = False
            self.play_button.setText("播放")
            self.record_button.setEnabled(True)
            self.load_button.setEnabled(True)
            self.playback_timer.stop()
            self.current_frame_index = 0
            
            logger.info("动作播放完成")
    
    def save_recording(self):
        """
        保存录制的动作到文件
        """
        if not hasattr(self, 'recorded_frames') or not self.recorded_frames:
            logger.warning("没有可保存的录制动作")
            return
        
        # 打开文件对话框选择保存位置
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存录制动作", ".", "动作文件 (*.json);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            import json
            
            # 转换录制数据为可序列化格式
            save_data = {
                'frame_count': len(self.recorded_frames),
                'frames': [frame.tolist() for frame in self.recorded_frames]
            }
            
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"动作录制已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存动作录制失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def mark_keyframe(self):
        """
        标记当前帧为关键帧
        """
        if not hasattr(self, 'recorded_frames') or not self.recorded_frames:
            logger.warning("没有可标记的录制动作")
            return
            
        # 获取当前帧索引
        if hasattr(self, 'is_playing') and self.is_playing and hasattr(self, 'current_frame_index'):
            frame_index = self.current_frame_index
        else:
            frame_index = len(self.recorded_frames) - 1
            
        if frame_index < 0 or frame_index >= len(self.recorded_frames):
            return
            
        # 添加关键帧
        if frame_index not in self.keyframes:
            self.keyframes.append(frame_index)
            self.keyframes.sort()
            
            # 更新关键帧列表
            self.keyframe_list.clear()
            for i, keyframe_idx in enumerate(self.keyframes):
                self.keyframe_list.addItem(f"关键帧 {i+1}: 第 {keyframe_idx+1} 帧")
                
            logger.info(f"标记关键帧: 第 {frame_index+1} 帧")
    
    def _jump_to_keyframe(self, item):
        """
        跳转到指定的关键帧
        """
        # 解析关键帧信息
        item_text = item.text()
        frame_index = int(item_text.split("第 ")[1].split(" 帧")[0]) - 1
        
        if frame_index >= 0 and frame_index < len(self.recorded_frames):
            # 更新当前帧索引
            if hasattr(self, 'current_frame_index'):
                self.current_frame_index = frame_index
            else:
                self.current_frame_index = frame_index
                
            # 更新人体模型
            frame_data = self.recorded_frames[frame_index]
            self.update_human_model(frame_data)
            
            logger.info(f"跳转到关键帧: 第 {frame_index+1} 帧")
    
    def save_recording(self):
        """
        保存录制的动作到文件
        """
        if not hasattr(self, 'recorded_frames') or not self.recorded_frames:
            logger.warning("没有可保存的录制动作")
            return
        
        # 打开文件对话框选择保存位置
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存录制动作", ".", "动作文件 (*.json);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            import json
            
            # 转换录制数据为可序列化格式
            save_data = {
                'frame_count': len(self.recorded_frames),
                'frames': [frame.tolist() for frame in self.recorded_frames],
                'keyframes': self.keyframes if hasattr(self, 'keyframes') else []
            }
            
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"动作录制已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存动作录制失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def load_recording(self):
        """
        从文件加载录制的动作
        """
        # 打开文件对话框选择加载文件
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载录制动作", ".", "动作文件 (*.json);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            import json
            import numpy as np
            
            with open(file_path, 'r') as f:
                load_data = json.load(f)
            
            # 解析录制数据
            self.recorded_frames = [np.array(frame) for frame in load_data['frames']]
            
            # 加载关键帧
            if 'keyframes' in load_data:
                self.keyframes = load_data['keyframes']
                # 更新关键帧列表
                self.keyframe_list.clear()
                for i, keyframe_idx in enumerate(self.keyframes):
                    self.keyframe_list.addItem(f"关键帧 {i+1}: 第 {keyframe_idx+1} 帧")
            else:
                self.keyframes = []
                self.keyframe_list.clear()
            
            logger.info(f"动作录制已从 {file_path} 加载，共 {len(self.recorded_frames)} 帧")
            
            # 启用播放和保存按钮
            self.play_button.setEnabled(len(self.recorded_frames) > 0)
            self.save_button.setEnabled(len(self.recorded_frames) > 0)
            self.keyframe_button.setEnabled(len(self.recorded_frames) > 0)
            
            # 如果已经导入了标准动作，启用比对按钮
            if hasattr(self, 'standard_frames') and self.standard_frames:
                self.compare_button.setEnabled(True)
                
        except Exception as e:
            logger.error(f"加载动作录制失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def load_standard_action(self):
        """
        从文件加载标准动作（用于比对）
        """
        # 打开文件对话框选择加载文件
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载标准动作", ".", "动作文件 (*.json);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            import json
            import numpy as np
            
            with open(file_path, 'r') as f:
                load_data = json.load(f)
            
            # 解析标准动作数据
            self.standard_frames = [np.array(frame) for frame in load_data['frames']]
            
            logger.info(f"标准动作已从 {file_path} 加载，共 {len(self.standard_frames)} 帧")
            
            # 如果已经加载了录制动作，启用比对按钮
            if hasattr(self, 'recorded_frames') and self.recorded_frames:
                self.compare_button.setEnabled(True)
                
        except Exception as e:
            logger.error(f"加载标准动作失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def start_comparison(self):
        """
        开始动作比对
        """
        if not hasattr(self, 'recorded_frames') or not self.recorded_frames:
            logger.warning("没有可比对的录制动作")
            return
            
        if not hasattr(self, 'standard_frames') or not self.standard_frames:
            logger.warning("没有可比对的标准动作")
            return
            
        logger.info("开始动作比对...")
        
        # 这里将在后续实现完整的比对算法
        # 当前仅作基础框架，显示基本信息
        logger.info(f"录制动作帧数: {len(self.recorded_frames)}")
        logger.info(f"标准动作帧数: {len(self.standard_frames)}")
        
        # 初始化比对结果数据结构
        self.comparison_results = {
            'frame_comparisons': [],
            'joint_analysis': {},
            'overall_score': 0.0
        }
        
        logger.info("动作比对基础框架已初始化")
    
    def calculate_joint_ranges(self):
        """
        计算并显示各关节的活动范围统计数据
        """
        # 清空表格
        self.range_table.setRowCount(0)
        
        if not self.angle_curves:
            logger.warning("没有关节角度数据可用于计算活动范围")
            return
            
        logger.info("开始计算关节活动范围...")
        
        # 遍历所有关节的角度数据
        for joint_name, data in self.angle_curves.items():
            if not data['values']:
                continue
                
            # 计算统计值
            min_angle = min(data['values'])
            max_angle = max(data['values'])
            range_angle = max_angle - min_angle
            
            # 添加到表格
            row = self.range_table.rowCount()
            self.range_table.insertRow(row)
            
            # 关节名称
            self.range_table.setItem(row, 0, QtWidgets.QTableWidgetItem(joint_name))
            
            # 最小角度
            min_item = QtWidgets.QTableWidgetItem(f"{min_angle:.2f}")
            min_item.setTextAlignment(QtCore.Qt.AlignRight)
            self.range_table.setItem(row, 1, min_item)
            
            # 最大角度
            max_item = QtWidgets.QTableWidgetItem(f"{max_angle:.2f}")
            max_item.setTextAlignment(QtCore.Qt.AlignRight)
            self.range_table.setItem(row, 2, max_item)
            
            # 活动范围
            range_item = QtWidgets.QTableWidgetItem(f"{range_angle:.2f}")
            range_item.setTextAlignment(QtCore.Qt.AlignRight)
            self.range_table.setItem(row, 3, range_item)
        
        # 调整列宽以适应内容
        self.range_table.resizeColumnsToContents()
        
        logger.info("关节活动范围计算完成")
    
    def clear_joint_ranges(self):
        """
        清除关节活动范围统计数据
        """
        self.range_table.setRowCount(0)
        logger.info("关节活动范围统计数据已清除")
    
    def select_color(self):
        """
        选择标注颜色
        """
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            r, g, b, a = color.getRgbF()
            self.current_color = (r, g, b, 1.0)  # PyQtGraph使用0-1范围的浮点数
            self.color_button.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});")
    
    def add_annotation(self):
        """
        添加标注到3D视图
        """
        annotation_type = self.annotation_type_combo.currentText()
        color = self.current_color
        width = self.width_spinbox.value()
        
        if annotation_type == "线条":
            # 创建线条标注
            # 这里简单实现为从原点到(5,5,5)的示例线条
            # 实际应用中可以通过鼠标选择起点和终点
            pos = np.array([[0, 0, 0], [5, 5, 5]])
            line = gl.GLLinePlotItem(pos=pos, color=color, width=width)
            self.view_3d.addItem(line)
            self.annotations.append(line)
            logger.info("已添加线条标注")
            
        elif annotation_type == "箭头":
            # 创建箭头标注
            # 从原点指向(5,5,5)
            pos = np.array([0, 0, 0])
            vec = np.array([5, 5, 5])
            
            # 使用GLLinePlotItem绘制箭头主线
            arrow_line = gl.GLLinePlotItem(pos=np.array([pos, pos+vec]), color=color, width=width)
            self.view_3d.addItem(arrow_line)
            
            # 计算箭头头部
            arrow_length = np.linalg.norm(vec) * 0.1
            arrow_angle = np.pi / 6  # 30度
            
            # 生成箭头头部的三个点
            vec_normalized = vec / np.linalg.norm(vec)
            # 创建垂直于vec的向量
            if np.allclose(vec_normalized, [1, 0, 0]):
                perp = np.array([0, 1, 0])
            else:
                perp = np.cross(vec_normalized, [1, 0, 0])
                perp /= np.linalg.norm(perp)
            
            perp2 = np.cross(vec_normalized, perp)
            
            arrow_tip = pos + vec
            arrow_base = arrow_tip - vec_normalized * arrow_length
            
            arrow_head1 = arrow_base + perp * arrow_length * np.tan(arrow_angle)
            arrow_head2 = arrow_base - perp * arrow_length * np.tan(arrow_angle)
            arrow_head3 = arrow_base + perp2 * arrow_length * np.tan(arrow_angle)
            arrow_head4 = arrow_base - perp2 * arrow_length * np.tan(arrow_angle)
            
            # 绘制箭头头部
            arrow_head = gl.GLLinePlotItem(
                pos=np.array([arrow_tip, arrow_head1, arrow_tip, arrow_head2, arrow_tip, arrow_head3, arrow_tip, arrow_head4]),
                color=color, width=width
            )
            self.view_3d.addItem(arrow_head)
            
            self.annotations.extend([arrow_line, arrow_head])
            logger.info("已添加箭头标注")
            
        elif annotation_type == "文字":
            # 创建文字标注
            text = self.text_input.text()
            font_size = self.font_size_spinbox.value()
            
            # 这里简单实现为在(5,5,5)位置添加文字
            # 注意：PyQtGraph的GLTextItem功能有限，这里使用GLLinePlotItem模拟简单文字
            # 实际应用中可能需要使用更高级的文字渲染方法
            logger.info(f"已添加文字标注: {text}")
            
            # 简单示例：在3D空间中显示文字位置标记
            pos = np.array([[5, 5, 5]])
            text_marker = gl.GLScatterPlotItem(pos=pos, color=color, size=font_size/2)
            self.view_3d.addItem(text_marker)
            self.annotations.append(text_marker)
    
    def clear_all_annotations(self):
        """
        清除所有标注
        """
        for annotation in self.annotations:
            self.view_3d.removeItem(annotation)
        self.annotations.clear()
        logger.info("已清除所有标注")
    
    def update(self):
        """
        更新可视化界面
        """
        # 使用更高效的方式处理事件
        if hasattr(self, 'app'):
            # 限制事件处理频率
            self.frame_count = getattr(self, 'frame_count', 0)
            self.frame_count += 1
            
            # 每2帧处理一次事件，减少CPU占用
            if self.frame_count % 2 == 0:
                self.app.processEvents()
                
            # 定期清理内存
            if self.frame_count % 100 == 0:
                import gc
                gc.collect()
    
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
            preset_name: 视图名称 ('front', 'side', 'top', 'back')
        """
        if preset_name == 'front':
            self.view_3d.setCameraPosition(distance=10, elevation=90, azimuth=0)
        elif preset_name == 'side':
            self.view_3d.setCameraPosition(distance=10, elevation=90, azimuth=90)
        elif preset_name == 'top':
            self.view_3d.setCameraPosition(distance=10, elevation=0, azimuth=0)
        elif preset_name == 'back':
            self.view_3d.setCameraPosition(distance=10, elevation=90, azimuth=180)
    
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