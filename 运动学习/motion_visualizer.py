import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

class MotionVisualizer:
    """
    运动数据可视化器，用于展示3D人体模型和运动参数图表
    """
    def __init__(self, app=None):
        """
        初始化可视化器
        
        参数:
            app: PyQt应用实例
        """
        if app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        else:
            self.app = app
        
        # 创建主窗口
        self.main_window = QtWidgets.QMainWindow()
        self.main_window.setWindowTitle("3D运动分析可视化")
        self.main_window.setGeometry(100, 100, 1200, 800)
        
        # 创建中心 widget 和布局
        self.central_widget = QtWidgets.QWidget()
        self.main_window.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        
        # 创建左侧3D视图
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
        
        # 显示窗口
        self.main_window.show()
    
    def setup_3d_view(self):
        """
        设置3D视图的背景和坐标轴
        """
        # 设置黑色背景
        self.view_3d.setBackgroundColor('k')
        
        # 添加坐标轴
        g = gl.GLGridItem()
        g.setSize(x=20, y=20, z=20)
        g.setSpacing(x=1, y=1, z=1)
        self.view_3d.addItem(g)
        
        # 添加坐标系
        ax = gl.GLAxisItem()
        ax.setSize(x=5, y=5, z=5)
        self.view_3d.addItem(ax)
        
        # 设置相机位置
        self.view_3d.setCameraPosition(distance=10, elevation=30, azimuth=45)
    
    def setup_charts(self):
        """
        创建运动参数图表
        """
        # 创建角度图表
        self.angle_plot_widget = pg.PlotWidget(title="关节角度")
        self.angle_plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.control_layout.addWidget(self.angle_plot_widget)
        
        # 创建速度图表
        self.velocity_plot_widget = pg.PlotWidget(title="关节速度")
        self.velocity_plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.control_layout.addWidget(self.velocity_plot_widget)
        
        # 创建加速度图表
        self.acceleration_plot_widget = pg.PlotWidget(title="关节加速度")
        self.acceleration_plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
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
        # 创建关节点
        self.joint_points = gl.GLScatterPlotItem(
            pos=np.zeros((33, 3)),  # 33个关节点
            size=0.1,
            color=(1, 0, 0, 1),
            pxMode=True
        )
        self.view_3d.addItem(self.joint_points)
        
        # 创建骨骼连接
        self.bone_lines = gl.GLLinePlotItem(
            pos=np.zeros((0, 3)),  # 初始无连接
            color=(0, 1, 0, 1),
            width=2,
            mode='lines'
        )
        self.view_3d.addItem(self.bone_lines)
        
        # 定义骨骼连接
        self.bone_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # 左眼
            (0, 4), (4, 5), (5, 6), (6, 8),  # 右眼
            (0, 9), (0, 10),  # 嘴巴
            (11, 12),  # 肩部连接
            (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),  # 左手臂
            (15, 21), (15, 19), (15, 17),  # 左手
            (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),  # 右手臂
            (16, 22), (16, 20), (16, 18),  # 右手
            (11, 23), (12, 24), (23, 24),  # 躯干
            (23, 25), (25, 27), (27, 29), (29, 31),  # 左腿
            (27, 31),  # 左脚
            (24, 26), (26, 28), (28, 30), (30, 32),  # 右腿
            (28, 32)   # 右脚
        ]
    
    def update_human_model(self, landmarks):
        """
        更新3D人体模型的关节点和骨骼位置
        
        参数:
            landmarks: 关节点坐标列表
        """
        if not landmarks:
            return
        
        # 更新关节点位置
        positions = np.array(landmarks)
        self.joint_points.setData(pos=positions)
        
        # 更新骨骼连接
        bone_positions = []
        for connection in self.bone_connections:
            try:
                start_joint = landmarks[connection[0]]
                end_joint = landmarks[connection[1]]
                bone_positions.append(start_joint)
                bone_positions.append(end_joint)
            except IndexError:
                continue
        
        if bone_positions:
            self.bone_lines.setData(pos=np.array(bone_positions))
    
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
        max_points = 100
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
        max_points = 100
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
        max_points = 100
        if len(self.acceleration_curves[joint_name]['data']) > max_points:
            self.acceleration_curves[joint_name]['data'] = self.acceleration_curves[joint_name]['data'][-max_points:]
    
    def get_color_for_joint(self, joint_name):
        """
        根据关节名称获取颜色
        
        参数:
            joint_name: 关节名称
            
        返回:
            color: 颜色元组 (r, g, b)
        """
        color_map = {
            'left_elbow': (1, 0, 0),
            'right_elbow': (0, 1, 0),
            'left_knee': (0, 0, 1),
            'right_knee': (1, 1, 0),
            'left_shoulder': (1, 0, 1),
            'right_shoulder': (0, 1, 1)
        }
        
        return color_map.get(joint_name, (0.5, 0.5, 0.5))
    
    def update(self):
        """
        更新可视化界面
        """
        self.app.processEvents()
    
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