import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    """
    AI姿态检测器，使用MediaPipe库实现人体关键关节点识别
    """
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        初始化姿态检测器
        
        参数:
            static_image_mode: 是否为静态图像模式
            model_complexity: 模型复杂度(0-2)
            smooth_landmarks: 是否平滑关节点
            enable_segmentation: 是否启用分割
            smooth_segmentation: 是否平滑分割
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化MediaPipe Pose模型
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect_pose(self, image, draw=True):
        """
        检测图像中的人体姿态
        
        参数:
            image: 输入图像 (BGR格式)
            draw: 是否在图像上绘制关节点
            
        返回:
            results: 姿态检测结果
            image: 处理后的图像
        """
        # 转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # 检测姿态
        results = self.pose.process(image_rgb)
        
        # 绘制关节点
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if draw and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return results, image
    
    def get_landmarks(self, results, mode='3d'):
        """
        获取检测到的关节点坐标
        
        参数:
            results: 姿态检测结果
            mode: 返回模式 ('2d' 或 '3d')
            
        返回:
            landmarks: 关节点坐标列表
        """
        if not results.pose_landmarks:
            return None
        
        landmarks = []
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            if mode == '2d':
                # 返回2D坐标 (x, y)
                landmarks.append((landmark.x, landmark.y))
            elif mode == '3d':
                # 返回3D坐标 (x, y, z)
                landmarks.append((landmark.x, landmark.y, landmark.z))
        
        return landmarks
    
    def get_joint_names(self):
        """
        获取关节点名称列表
        
        返回:
            joint_names: 关节点名称列表
        """
        return [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear',
            'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
            'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
            'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
            'right_index', 'left_thumb', 'right_thumb', 'left_hip',
            'right_hip', 'left_knee', 'right_knee', 'left_ankle',
            'right_ankle', 'left_heel', 'right_heel', 'left_foot_index',
            'right_foot_index'
        ]
    
    def close(self):
        """
        关闭姿态检测器
        """
        self.pose.close()

if __name__ == "__main__":
    # 测试姿态检测器
    cap = cv2.VideoCapture(0)  # 打开摄像头
    detector = PoseDetector()
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("无法读取摄像头帧")
            break
        
        # 检测姿态
        results, image = detector.detect_pose(image)
        
        # 获取关节点
        landmarks_3d = detector.get_landmarks(results, mode='3d')
        if landmarks_3d:
            print(f"检测到 {len(landmarks_3d)} 个关节点")
        
        # 显示结果
        cv2.imshow('Pose Detection', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    detector.close()
    cap.release()
    cv2.destroyAllWindows()