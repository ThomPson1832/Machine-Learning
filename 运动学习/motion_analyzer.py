import numpy as np
from scipy.signal import butter, filtfilt

class MotionAnalyzer:
    """
    3D运动分析器，用于计算人体运动的各项参数
    """
    def __init__(self, fps=30):
        """
        初始化运动分析器
        
        参数:
            fps: 视频帧率
        """
        self.fps = fps
        self.frame_time = 1.0 / fps
        
        # 存储历史关节点数据
        self.landmarks_history = []
        
        # 关节点索引映射
        self.joint_indices = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
    
    def add_frame(self, landmarks):
        """
        添加一帧的关节点数据
        
        参数:
            landmarks: 当前帧的关节点坐标列表
        """
        if landmarks:
            self.landmarks_history.append(landmarks)
            # 限制历史数据长度，避免内存占用过大
            if len(self.landmarks_history) > 300:  # 保存10秒的数据
                self.landmarks_history.pop(0)
    
    def calculate_angle(self, p1, p2, p3):
        """
        计算三个点构成的角度
        
        参数:
            p1, p2, p3: 三个点的坐标 (x, y, z)
            
        返回:
            angle: 角度值 (度)
        """
        # 转换为numpy数组
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        
        # 计算向量
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 计算夹角
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免浮点误差
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        
        return angle
    
    def calculate_joint_angle(self, joint_name, frame_idx=None):
        """
        计算指定关节的角度
        
        参数:
            joint_name: 关节名称 (如 'left_elbow', 'right_knee')
            frame_idx: 帧索引，None表示当前帧
            
        返回:
            angle: 关节角度 (度)
        """
        if frame_idx is None:
            frame_idx = len(self.landmarks_history) - 1
        
        if frame_idx < 0 or frame_idx >= len(self.landmarks_history):
            return None
        
        landmarks = self.landmarks_history[frame_idx]
        
        try:
            if joint_name == 'left_elbow':
                # 左肘角度: 左肩 - 左肘 - 左腕
                return self.calculate_angle(
                    landmarks[self.joint_indices['left_shoulder']],
                    landmarks[self.joint_indices['left_elbow']],
                    landmarks[self.joint_indices['left_wrist']]
                )
            elif joint_name == 'right_elbow':
                # 右肘角度: 右肩 - 右肘 - 右腕
                return self.calculate_angle(
                    landmarks[self.joint_indices['right_shoulder']],
                    landmarks[self.joint_indices['right_elbow']],
                    landmarks[self.joint_indices['right_wrist']]
                )
            elif joint_name == 'left_knee':
                # 左膝角度: 左髋 - 左膝 - 左踝
                return self.calculate_angle(
                    landmarks[self.joint_indices['left_hip']],
                    landmarks[self.joint_indices['left_knee']],
                    landmarks[self.joint_indices['left_ankle']]
                )
            elif joint_name == 'right_knee':
                # 右膝角度: 右髋 - 右膝 - 右踝
                return self.calculate_angle(
                    landmarks[self.joint_indices['right_hip']],
                    landmarks[self.joint_indices['right_knee']],
                    landmarks[self.joint_indices['right_ankle']]
                )
            elif joint_name == 'left_shoulder':
                # 左肩角度: 右肩 - 左肩 - 左肘
                return self.calculate_angle(
                    landmarks[self.joint_indices['right_shoulder']],
                    landmarks[self.joint_indices['left_shoulder']],
                    landmarks[self.joint_indices['left_elbow']]
                )
            elif joint_name == 'right_shoulder':
                # 右肩角度: 左肩 - 右肩 - 右肘
                return self.calculate_angle(
                    landmarks[self.joint_indices['left_shoulder']],
                    landmarks[self.joint_indices['right_shoulder']],
                    landmarks[self.joint_indices['right_elbow']]
                )
            else:
                return None
        except (IndexError, ValueError):
            return None
    
    def calculate_velocity(self, joint_name):
        """
        计算指定关节点的速度
        
        参数:
            joint_name: 关节名称
            
        返回:
            velocity: 速度值 (单位/秒)
        """
        if len(self.landmarks_history) < 2:
            return None
        
        try:
            # 获取当前帧和前一帧的关节点坐标
            current_landmarks = self.landmarks_history[-1]
            prev_landmarks = self.landmarks_history[-2]
            
            joint_idx = self.joint_indices[joint_name]
            current_pos = np.array(current_landmarks[joint_idx])
            prev_pos = np.array(prev_landmarks[joint_idx])
            
            # 计算位移和速度
            displacement = np.linalg.norm(current_pos - prev_pos)
            velocity = displacement / self.frame_time
            
            return velocity
        except (IndexError, KeyError):
            return None
    
    def calculate_acceleration(self, joint_name):
        """
        计算指定关节点的加速度
        
        参数:
            joint_name: 关节名称
            
        返回:
            acceleration: 加速度值 (单位/秒²)
        """
        if len(self.landmarks_history) < 3:
            return None
        
        try:
            # 计算当前速度和前一速度
            current_velocity = self._get_velocity_at_frame(len(self.landmarks_history) - 1, joint_name)
            prev_velocity = self._get_velocity_at_frame(len(self.landmarks_history) - 2, joint_name)
            
            if current_velocity is None or prev_velocity is None:
                return None
            
            # 计算加速度
            acceleration = (current_velocity - prev_velocity) / self.frame_time
            
            return acceleration
        except Exception:
            return None
    
    def _get_velocity_at_frame(self, frame_idx, joint_name):
        """
        获取指定帧的关节点速度
        
        参数:
            frame_idx: 帧索引
            joint_name: 关节名称
            
        返回:
            velocity: 速度值
        """
        if frame_idx < 1 or frame_idx >= len(self.landmarks_history):
            return None
        
        try:
            current_landmarks = self.landmarks_history[frame_idx]
            prev_landmarks = self.landmarks_history[frame_idx - 1]
            
            joint_idx = self.joint_indices[joint_name]
            current_pos = np.array(current_landmarks[joint_idx])
            prev_pos = np.array(prev_landmarks[joint_idx])
            
            displacement = np.linalg.norm(current_pos - prev_pos)
            velocity = displacement / self.frame_time
            
            return velocity
        except (IndexError, KeyError):
            return None
    
    def smooth_data(self, data, cutoff=5.0):
        """
        对数据进行平滑处理
        
        参数:
            data: 原始数据列表
            cutoff: 截止频率
            
        返回:
            smoothed_data: 平滑后的数据列表
        """
        if len(data) < 3:
            return data
        
        # 设计巴特沃斯低通滤波器
        nyq = 0.5 * self.fps
        normal_cutoff = cutoff / nyq
        b, a = butter(3, normal_cutoff, btype='low', analog=False)
        
        # 应用滤波器
        smoothed_data = filtfilt(b, a, data)
        
        return smoothed_data.tolist()
    
    def get_joint_trajectory(self, joint_name, smooth=True):
        """
        获取指定关节点的运动轨迹
        
        参数:
            joint_name: 关节名称
            smooth: 是否平滑轨迹
            
        返回:
            trajectory: 关节点轨迹数据
        """
        if not self.landmarks_history:
            return None
        
        try:
            joint_idx = self.joint_indices[joint_name]
            trajectory = [landmarks[joint_idx] for landmarks in self.landmarks_history]
            
            if smooth:
                # 分别对x, y, z坐标进行平滑
                x_coords = [p[0] for p in trajectory]
                y_coords = [p[1] for p in trajectory]
                z_coords = [p[2] for p in trajectory]
                
                x_coords_smooth = self.smooth_data(x_coords)
                y_coords_smooth = self.smooth_data(y_coords)
                z_coords_smooth = self.smooth_data(z_coords)
                
                trajectory = list(zip(x_coords_smooth, y_coords_smooth, z_coords_smooth))
            
            return trajectory
        except (IndexError, KeyError):
            return None
    
    def clear_history(self):
        """
        清除历史数据
        """
        self.landmarks_history.clear()

if __name__ == "__main__":
    # 测试运动分析器
    analyzer = MotionAnalyzer(fps=30)
    
    # 模拟关节点数据
    import random
    
    for i in range(100):
        # 生成随机关节点数据
        landmarks = []
        for j in range(33):  # 33个关节点
            x = 0.5 + random.uniform(-0.2, 0.2)
            y = 0.5 + random.uniform(-0.2, 0.2)
            z = 0.5 + random.uniform(-0.2, 0.2)
            landmarks.append((x, y, z))
        
        analyzer.add_frame(landmarks)
    
    # 测试角度计算
    left_elbow_angle = analyzer.calculate_joint_angle('left_elbow')
    print(f"左肘角度: {left_elbow_angle:.2f}度")
    
    # 测试速度计算
    left_wrist_velocity = analyzer.calculate_velocity('left_wrist')
    print(f"左手腕速度: {left_wrist_velocity:.4f}单位/秒")
    
    # 测试轨迹获取
    trajectory = analyzer.get_joint_trajectory('right_knee')
    if trajectory:
        print(f"右膝轨迹点数量: {len(trajectory)}")