import numpy as np
from scipy.signal import butter, filtfilt
import logging

# 设置日志记录
logger = logging.getLogger(__name__)

class MotionAnalyzer:
    """
    3D运动分析器，用于计算人体运动的各项参数
    """
    def __init__(self, fps=30, max_history_length=300):
        """
        初始化运动分析器
        
        参数:
            fps: 视频帧率
            max_history_length: 最大历史数据长度（默认300帧，约10秒）
        """
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.max_history_length = max_history_length
        
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
        
        # 关节角度计算配置（使用字典映射替代if-elif分支，提高性能）
        self.joint_angle_config = {
            # 基础关节角度计算
            'left_elbow': lambda lm: self.calculate_angle(
                lm[self.joint_indices['left_shoulder']],
                lm[self.joint_indices['left_elbow']],
                lm[self.joint_indices['left_wrist']]
            ),
            'right_elbow': lambda lm: self.calculate_angle(
                lm[self.joint_indices['right_shoulder']],
                lm[self.joint_indices['right_elbow']],
                lm[self.joint_indices['right_wrist']]
            ),
            'left_knee': lambda lm: self.calculate_angle(
                lm[self.joint_indices['left_hip']],
                lm[self.joint_indices['left_knee']],
                lm[self.joint_indices['left_ankle']]
            ),
            'right_knee': lambda lm: self.calculate_angle(
                lm[self.joint_indices['right_hip']],
                lm[self.joint_indices['right_knee']],
                lm[self.joint_indices['right_ankle']]
            ),
            'left_shoulder': lambda lm: self.calculate_angle(
                lm[self.joint_indices['right_shoulder']],
                lm[self.joint_indices['left_shoulder']],
                lm[self.joint_indices['left_elbow']]
            ),
            'right_shoulder': lambda lm: self.calculate_angle(
                lm[self.joint_indices['left_shoulder']],
                lm[self.joint_indices['right_shoulder']],
                lm[self.joint_indices['right_elbow']]
            ),
            'left_hip': lambda lm: self.calculate_angle(
                lm[self.joint_indices['right_hip']],
                lm[self.joint_indices['left_hip']],
                lm[self.joint_indices['left_knee']]
            ),
            'right_hip': lambda lm: self.calculate_angle(
                lm[self.joint_indices['left_hip']],
                lm[self.joint_indices['right_hip']],
                lm[self.joint_indices['right_knee']]
            ),
            
            # 需要虚拟点的关节角度计算
            'left_ankle': lambda lm: self._calculate_angle_with_extension(
                lm[self.joint_indices['left_knee']],
                lm[self.joint_indices['left_ankle']],
                extension_factor=0.5
            ),
            'right_ankle': lambda lm: self._calculate_angle_with_extension(
                lm[self.joint_indices['right_knee']],
                lm[self.joint_indices['right_ankle']],
                extension_factor=0.5
            ),
            'left_wrist': lambda lm: self._calculate_angle_with_extension(
                lm[self.joint_indices['left_elbow']],
                lm[self.joint_indices['left_wrist']],
                extension_factor=0.5
            ),
            'right_wrist': lambda lm: self._calculate_angle_with_extension(
                lm[self.joint_indices['right_elbow']],
                lm[self.joint_indices['right_wrist']],
                extension_factor=0.5
            ),
            
            # 肩部旋转角度
            'left_shoulder_rotation': lambda lm: self.calculate_angle(
                lm[self.joint_indices['right_shoulder']],
                lm[self.joint_indices['left_shoulder']],
                lm[self.joint_indices['left_wrist']]
            ),
            'right_shoulder_rotation': lambda lm: self.calculate_angle(
                lm[self.joint_indices['left_shoulder']],
                lm[self.joint_indices['right_shoulder']],
                lm[self.joint_indices['right_wrist']]
            )
        }
        
        logger.info(f"运动分析器初始化完成，帧率: {fps}, 最大历史长度: {max_history_length}")
    
    def add_frame(self, landmarks):
        """
        添加一帧的关节点数据
        
        参数:
            landmarks: 当前帧的关节点坐标列表
        """
        try:
            if landmarks:
                self.landmarks_history.append(landmarks)
                # 限制历史数据长度，避免内存占用过大
                if len(self.landmarks_history) > self.max_history_length:
                    self.landmarks_history.pop(0)
                
                # 定期记录历史数据大小
                # 减少日志输出频率，每500帧记录一次
                if len(self.landmarks_history) % 500 == 0:
                    logger.debug(f"历史数据大小: {len(self.landmarks_history)} 帧")
        except Exception as e:
            logger.error(f"添加帧数据失败: {str(e)}")
    
    def calculate_angle(self, p1, p2, p3):
        """
        计算三个点构成的角度（优化版）
        
        参数:
            p1, p2, p3: 三个点的坐标 (x, y, z)
            
        返回:
            angle: 角度值 (度)
        """
        try:
            # 直接使用列表操作避免numpy数组转换开销
            v1x = p1[0] - p2[0]
            v1y = p1[1] - p2[1]
            v1z = p1[2] - p2[2]
            
            v2x = p3[0] - p2[0]
            v2y = p3[1] - p2[1]
            v2z = p3[2] - p2[2]
            
            # 计算点积
            dot_product = v1x * v2x + v1y * v2y + v1z * v2z
            
            # 计算模长
            norm1 = (v1x ** 2 + v1y ** 2 + v1z ** 2) ** 0.5
            norm2 = (v2x ** 2 + v2y ** 2 + v2z ** 2) ** 0.5
            
            # 避免除以零
            if norm1 < 1e-8 or norm2 < 1e-8:
                return None
            
            # 计算夹角余弦
            cos_angle = dot_product / (norm1 * norm2)
            # 避免浮点误差
            cos_angle = max(-1.0, min(1.0, cos_angle))
            
            # 转换为角度
            angle = np.arccos(cos_angle) * 180.0 / np.pi
            
            return angle
        except Exception:
            # 减少日志记录，提高性能
            return 0.0
    
    def _calculate_angle_with_extension(self, point1, point2, extension_factor=0.5):
        """
        计算带有延伸点的角度（用于计算需要虚拟点的关节角度，如脚踝、手腕）
        
        参数:
            point1: 第一个点 (x, y, z)
            point2: 第二个点 (x, y, z)
            extension_factor: 延伸因子
            
        返回:
            angle: 角度值 (度)
        """
        try:
            # 计算从point1到point2的方向向量
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            dz = point2[2] - point1[2]
            
            # 计算延伸点（从point2向前延伸）
            extended_point = (
                point2[0] + dx * extension_factor,
                point2[1] + dy * extension_factor,
                point2[2] + dz * extension_factor
            )
            
            # 计算角度
            return self.calculate_angle(point1, point2, extended_point)
        except Exception as e:
            logger.error(f"计算延伸角度失败: {str(e)}")
            return None

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
            logger.debug(f"无效的帧索引: {frame_idx}")
            return None
        
        landmarks = self.landmarks_history[frame_idx]
        
        try:
            if joint_name in self.joint_angle_config:
                return self.joint_angle_config[joint_name](landmarks)
            else:
                logger.warning(f"未知的关节名称: {joint_name}")
                return None
        except (IndexError, ValueError) as e:
            logger.error(f"计算关节角度失败 ({joint_name}): {str(e)}")
            return None
    
    def _calculate_velocity_between_frames(self, joint_name, frame1_idx, frame2_idx):
        """
        计算两个帧之间的关节点速度（内部辅助方法）
        
        参数:
            joint_name: 关节名称
            frame1_idx: 第一帧索引
            frame2_idx: 第二帧索引
            
        返回:
            velocity: 速度值 (单位/秒)
        """
        try:
            # 获取两帧的关节点坐标
            landmarks1 = self.landmarks_history[frame1_idx]
            landmarks2 = self.landmarks_history[frame2_idx]
            
            joint_idx = self.joint_indices[joint_name]
            pos1 = landmarks1[joint_idx]
            pos2 = landmarks2[joint_idx]
            
            # 计算位移（欧几里得距离）
            displacement = ((pos1[0] - pos2[0]) ** 2 + 
                           (pos1[1] - pos2[1]) ** 2 + 
                           (pos1[2] - pos2[2]) ** 2) ** 0.5
            
            # 计算速度
            time_diff = abs(frame2_idx - frame1_idx) * self.frame_time
            velocity = displacement / time_diff if time_diff > 0 else 0
            
            return velocity
        except (IndexError, KeyError) as e:
            logger.error(f"计算两帧间速度失败 ({joint_name}, 帧: {frame1_idx}-{frame2_idx}): {str(e)}")
            return None

    def calculate_velocity(self, joint_name, smooth=True, window_size=3):
        """
        计算指定关节点的速度，支持平滑处理
        
        参数:
            joint_name: 关节名称
            smooth: 是否平滑速度数据
            window_size: 平滑窗口大小
            
        返回:
            velocity: 速度值 (单位/秒)
        """
        if len(self.landmarks_history) < 2:
            logger.debug("历史数据不足，无法计算速度")
            return None
        
        if not smooth or len(self.landmarks_history) < window_size:
                # 确保历史数据足够计算速度
                if len(self.landmarks_history) < 2:
                    logger.debug("历史数据不足，无法计算速度")
                    return None
                # 计算当前帧与前一帧的速度
                return self._calculate_velocity_between_frames(
                    joint_name, len(self.landmarks_history) - 2, len(self.landmarks_history) - 1
                )
        else:
            # 使用滑动窗口平滑速度
            velocities = []
            for i in range(len(self.landmarks_history) - 1):
                vel = self._calculate_velocity_between_frames(
                    joint_name, i, i + 1
                )
                if vel is not None:
                    velocities.append(vel)
            
            if len(velocities) < window_size:
                return velocities[-1] if velocities else None
            
            # 计算最近window_size个速度的加权平均
            weights = np.arange(1, window_size + 1)
            weights = weights / np.sum(weights)
            recent_velocities = velocities[-window_size:]
            
            # 异常值检测与过滤
            mean_vel = np.mean(recent_velocities)
            std_vel = np.std(recent_velocities)
            filtered_velocities = [v for v in recent_velocities if abs(v - mean_vel) < 2 * std_vel]
            
            if not filtered_velocities:
                return recent_velocities[-1]
            
            return np.average(filtered_velocities, weights=weights[-len(filtered_velocities):])

    def calculate_acceleration(self, joint_name, smooth=True, window_size=3):
        """
        计算指定关节点的加速度，支持平滑处理
        
        参数:
            joint_name: 关节名称
            smooth: 是否平滑加速度数据
            window_size: 平滑窗口大小
            
        返回:
            acceleration: 加速度值 (单位/秒²)
        """
        if len(self.landmarks_history) < 3:
            logger.debug("历史数据不足，无法计算加速度")
            return None
        
        try:
            if not smooth or len(self.landmarks_history) < window_size + 1:
                # 获取最近三帧的索引
                current_idx = len(self.landmarks_history) - 1
                prev_idx = current_idx - 1
                prev_prev_idx = prev_idx - 1
                
                # 计算连续两帧的速度
                vel1 = self._calculate_velocity_between_frames(joint_name, prev_prev_idx, prev_idx)
                vel2 = self._calculate_velocity_between_frames(joint_name, prev_idx, current_idx)
                
                if vel1 is None or vel2 is None:
                    logger.debug("无法获取速度数据，无法计算加速度")
                    return None
                
                # 计算加速度
                acceleration = (vel2 - vel1) / self.frame_time
                return acceleration
            else:
                # 使用滑动窗口平滑加速度
                velocities = []
                for i in range(len(self.landmarks_history) - 1):
                    vel = self._calculate_velocity_between_frames(
                        joint_name, i, i + 1
                    )
                    if vel is not None:
                        velocities.append(vel)
                
                if len(velocities) < 2:
                    logger.debug("速度数据不足，无法计算加速度")
                    return None
                elif len(velocities) < window_size + 1:
                    vel1, vel2 = velocities[-2], velocities[-1]
                    return (vel2 - vel1) / self.frame_time
                
                # 计算加速度序列
                accelerations = []
                for i in range(1, len(velocities)):
                    acc = (velocities[i] - velocities[i-1]) / self.frame_time
                    accelerations.append(acc)
                
                if len(accelerations) < window_size:
                    return accelerations[-1] if accelerations else None
                
                # 计算最近window_size个加速度的加权平均
                weights = np.arange(1, window_size + 1)
                weights = weights / np.sum(weights)
                recent_accelerations = accelerations[-window_size:]
                
                # 异常值检测与过滤
                mean_acc = np.mean(recent_accelerations)
                std_acc = np.std(recent_accelerations)
                filtered_accelerations = [a for a in recent_accelerations if abs(a - mean_acc) < 2.5 * std_acc]
                
                if not filtered_accelerations:
                    return recent_accelerations[-1]
                
                return np.average(filtered_accelerations, weights=weights[-len(filtered_accelerations):])
        except Exception as e:
            logger.error(f"计算加速度失败 ({joint_name}): {str(e)}")
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
            logger.debug(f"无效的帧索引: {frame_idx}")
            return None
        
        # 计算指定帧与前一帧的速度
        return self._calculate_velocity_between_frames(
            joint_name, frame_idx - 1, frame_idx
        )
    
    def smooth_data(self, data, cutoff=5.0, method='butterworth'):
        """
        对数据进行平滑处理，支持多种平滑方法
        
        参数:
            data: 原始数据列表
            cutoff: 截止频率
            method: 平滑方法 ('butterworth', 'moving_average', 'savgol')
            
        返回:
            smoothed_data: 平滑后的数据列表
        """
        if len(data) < 3:
            logger.debug("数据点不足，无法平滑")
            return data
        
        try:
            if method == 'moving_average':
                # 移动平均平滑
                window_size = 5
                if len(data) < window_size:
                    window_size = 3
                weights = np.ones(window_size) / window_size
                smoothed_data = np.convolve(data, weights, mode='valid')
                # 保持数据长度一致
                pad_length = len(data) - len(smoothed_data)
                return np.pad(smoothed_data, (pad_length//2, pad_length - pad_length//2), 'edge').tolist()
            elif method == 'savgol':
                # 萨维茨基-戈雷滤波
                from scipy.signal import savgol_filter
                window_length = min(7, len(data))
                if window_length % 2 == 0:
                    window_length -= 1
                smoothed_data = savgol_filter(data, window_length, 2)
                return smoothed_data.tolist()
            else:  # butterworth
                # 设计巴特沃斯低通滤波器
                nyq = 0.5 * self.fps
                normal_cutoff = cutoff / nyq
                b, a = butter(3, normal_cutoff, btype='low', analog=False)
                
                # 应用滤波器
                smoothed_data = filtfilt(b, a, data)
                
                return smoothed_data.tolist()
        except ImportError:
            logger.warning("缺少必要的库，使用巴特沃斯滤波器")
            # 设计巴特沃斯低通滤波器
            nyq = 0.5 * self.fps
            normal_cutoff = cutoff / nyq
            b, a = butter(3, normal_cutoff, btype='low', analog=False)
            
            # 应用滤波器
            smoothed_data = filtfilt(b, a, data)
            
            return smoothed_data.tolist()
        except Exception as e:
            logger.error(f"数据平滑失败: {str(e)}")
            return data
    
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
            logger.debug("历史数据为空，无法获取轨迹")
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
            
            logger.debug(f"获取{joint_name}轨迹，包含{len(trajectory)}个点")
            return trajectory
        except (IndexError, KeyError) as e:
            logger.error(f"获取轨迹失败 ({joint_name}): {str(e)}")
            return None
    
    def clear_history(self):
        """
        清除历史数据
        """
        history_length = len(self.landmarks_history)
        self.landmarks_history.clear()
        logger.info(f"已清除历史数据，共 {history_length} 帧")
    
    def export_data(self, filename, start_time=0):
        """
        导出运动分析数据到CSV文件
        
        参数:
            filename: 导出的CSV文件名
            start_time: 起始时间（秒），用于计算相对时间
            
        返回:
            bool: 导出是否成功
        """
        try:
            if not self.landmarks_history:
                logger.warning("没有历史数据可导出")
                return False
            
            # 导入pandas（仅在需要时导入以减少启动时间）
            import pandas as pd
            import time
            
            # 准备导出数据
            data = []
            
            # 遍历每一帧数据
            for frame_idx, landmarks in enumerate(self.landmarks_history):
                # 计算时间戳（相对时间）
                timestamp = start_time + frame_idx / self.fps
                
                # 创建数据行
                row = {'timestamp': timestamp}
                
                # 计算所有支持的关节角度
                for joint_name in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee',
                                 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                                 'left_ankle', 'right_ankle', 'left_wrist', 'right_wrist']:
                    # 角度
                    angle = self.calculate_joint_angle(joint_name, frame_idx)
                    if angle is not None:
                        row[f'{joint_name}_angle'] = angle
                    
                    # 速度
                    velocity = self._get_velocity_at_frame(frame_idx, joint_name)
                    if velocity is not None:
                        row[f'{joint_name}_velocity'] = velocity
                    
                    # 加速度
                    # 计算加速度需要至少3帧数据
                    if frame_idx >= 2:
                        prev_velocity = self._get_velocity_at_frame(frame_idx - 1, joint_name)
                        if velocity is not None and prev_velocity is not None:
                            acceleration = (velocity - prev_velocity) * self.fps  # 转换为加速度（单位/秒²）
                            row[f'{joint_name}_acceleration'] = acceleration
                
                data.append(row)
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 导出到CSV
            df.to_csv(filename, index=False)
            
            logger.info(f"数据成功导出到: {filename}")
            logger.debug(f"导出数据行数: {len(df)}, 列数: {len(df.columns)}")
            
            return True
            
        except ImportError as e:
            logger.error(f"导出数据失败，缺少依赖: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"导出数据失败: {str(e)}")
            return False
    
    def get_history(self):
        """
        获取所有关节的历史数据（角度和速度）
        
        返回:
            history: 包含所有关节历史数据的字典
        """
        history = {}
        
        # 获取所有关节名称
        joint_names = list(self.joint_indices.keys())
        
        # 为每个关节收集历史数据
        for joint_name in joint_names:
            joint_history = {}
            
            # 遍历所有帧
            for frame_idx in range(len(self.landmarks_history)):
                # 获取时间戳
                timestamp = frame_idx / self.fps
                
                # 计算当前帧的角度和速度
                angle = self.calculate_joint_angle(joint_name, frame_idx)
                velocity = self._get_velocity_at_frame(frame_idx, joint_name)
                
                if angle is not None and velocity is not None:
                    joint_history[timestamp] = {
                        'angle': angle,
                        'speed': velocity
                    }
            
            history[joint_name] = joint_history
        
        return history

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