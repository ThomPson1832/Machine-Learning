# face_recognizer.py - 人脸识别模块（修复版）
import cv2
import numpy as np
import json
import os
import time
import logging
from typing import List, Tuple, Dict, Optional
import sklearn.metrics.pairwise as sk_metrics


class FaceRecognizer:
    """人脸识别器 - 修复版本"""

    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.face_database = {}
        self.face_detector = None
        self.recognition_threshold = self.config.get('recognition_threshold', 0.6)

        self.initialize_detector()
        self.load_face_database()

    def load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            return {}

    def initialize_detector(self):
        """初始化人脸检测器 - 修复版本"""
        try:
            # 尝试多个可能的Haar级联分类器路径
            cascade_paths = [
                # 标准OpenCV路径
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                # 备用路径
                'haarcascade_frontalface_default.xml',
                # 完整路径
                'C:/ProgramData/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml',
                # 用户路径
                os.path.join(os.path.expanduser('~'), '.conda', 'envs', '机器学习', 'Lib', 'site-packages', 'cv2',
                             'data', 'haarcascade_frontalface_default.xml')
            ]

            # 尝试加载级联分类器
            for cascade_path in cascade_paths:
                if os.path.exists(cascade_path):
                    self.face_detector = cv2.CascadeClassifier(cascade_path)
                    if not self.face_detector.empty():
                        logging.info(f"✅ 使用Haar级联人脸检测器: {cascade_path}")
                        return

            # 如果所有路径都失败，使用备用检测方法
            logging.warning("❌ 无法加载Haar级联分类器，将使用备用检测方法")
            self.face_detector = None

        except Exception as e:
            logging.error(f"人脸检测器初始化失败: {e}")
            self.face_detector = None

    def load_face_database(self):
        """加载人脸数据库"""
        db_path = self.config.get('face_database', 'face_database.json')
        try:
            if os.path.exists(db_path):
                with open(db_path, 'r', encoding='utf-8') as f:
                    self.face_database = json.load(f)
                logging.info(f"已加载 {len(self.face_database)} 个人脸数据")
            else:
                logging.info("未找到人脸数据库，将创建新数据库")
                self.face_database = {}
        except Exception as e:
            logging.error(f"加载人脸数据库失败: {e}")
            self.face_database = {}

    def save_face_database(self):
        """保存人脸数据库"""
        db_path = self.config.get('face_database', 'face_database.json')
        try:
            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(self.face_database, f, ensure_ascii=False, indent=2)
            logging.info("人脸数据库已保存")
        except Exception as e:
            logging.error(f"保存人脸数据库失败: {e}")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测人脸 - 支持多种检测方法"""
        try:
            # 首先尝试使用Haar级联分类器
            if self.face_detector is not None and not self.face_detector.empty():
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                if len(faces) > 0:
                    return [(x, y, w, h) for (x, y, w, h) in faces]

            # 如果Haar检测失败或不可用，使用备用方法
            return self.detect_faces_fallback(image)

        except Exception as e:
            logging.error(f"人脸检测失败: {e}")
            # 使用备用方法
            return self.detect_faces_fallback(image)

    def detect_faces_fallback(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """备用的人脸检测方法 - 基于肤色和轮廓"""
        try:
            faces = []

            # 方法1: 肤色检测
            faces_skin = self.detect_faces_skin_color(image)
            if faces_skin:
                faces.extend(faces_skin)

            # 方法2: 轮廓检测
            faces_contour = self.detect_faces_contour(image)
            if faces_contour:
                faces.extend(faces_contour)

            # 去重
            if len(faces) > 1:
                faces = self.merge_overlapping_faces(faces)

            return faces

        except Exception as e:
            logging.error(f"备用检测方法失败: {e}")
            return []

    def detect_faces_skin_color(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """基于肤色的人脸检测"""
        try:
            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 定义肤色范围
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # 创建肤色掩码
            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 高斯模糊
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            faces = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 50000:  # 合理的面积范围
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    # 人脸通常的宽高比
                    if 0.5 < aspect_ratio < 2.0:
                        # 稍微扩展区域
                        x = max(0, x - 15)
                        y = max(0, y - 15)
                        w = min(image.shape[1] - x, w + 30)
                        h = min(image.shape[0] - y, h + 30)
                        faces.append((x, y, w, h))

            return faces

        except Exception as e:
            logging.error(f"肤色检测失败: {e}")
            return []

    def detect_faces_contour(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """基于轮廓的人脸检测"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 自适应阈值
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

            # 形态学操作
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            faces = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1500 < area < 30000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    if 0.5 < aspect_ratio < 2.0:
                        faces.append((x, y, w, h))

            return faces
        except Exception as e:
            logging.error(f"轮廓检测失败: {e}")
            return []

    def merge_overlapping_faces(self, faces):
        """合并重叠的人脸框"""
        if len(faces) <= 1:
            return faces

        # 转换格式
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])

        boxes = np.array(boxes)

        # 简单的非最大抑制
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > 0.5)[0])))

        # 转换回原始格式
        merged_faces = []
        for i in pick:
            x1, y1, x2, y2 = boxes[i]
            w = x2 - x1
            h = y2 - y1
            merged_faces.append((x1, y1, w, h))

        return merged_faces

    def extract_face_features(self, face_roi: np.ndarray) -> np.ndarray:
        """提取人脸特征"""
        try:
            # 调整大小
            face_resized = cv2.resize(face_roi, (100, 100))

            # 转换为灰度
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

            features = []

            # 1. 直方图特征
            hist = cv2.calcHist([gray_face], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

            # 2. 边缘特征
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1]) if edges.size > 0 else 0
            features.append(edge_density)

            # 3. 纹理特征 - 简化的LBP
            lbp_features = self.simple_lbp(gray_face)
            features.extend(lbp_features)

            # 4. 颜色特征
            hsv_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
            color_hist = cv2.calcHist([hsv_face], [0, 1], None, [8, 8], [0, 180, 0, 256])
            color_hist = cv2.normalize(color_hist, color_hist).flatten()
            features.extend(color_hist)

            return np.array(features)

        except Exception as e:
            logging.error(f"特征提取失败: {e}")
            return np.zeros(64 + 1 + 16 + 64)

    def simple_lbp(self, gray_face: np.ndarray) -> np.ndarray:
        """简化的LBP特征"""
        try:
            # 将图像分割为4x4区域
            h, w = gray_face.shape
            features = []

            for i in range(4):
                for j in range(4):
                    roi = gray_face[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]
                    if roi.size > 0:
                        features.append(np.mean(roi))
                        features.append(np.std(roi))

            return np.array(features)
        except:
            return np.zeros(16)

    def recognize_face(self, features: np.ndarray) -> Tuple[str, float]:
        """识别人脸"""
        if not self.face_database:
            return "未知人员", 1.0

        best_match = "未知人员"
        best_similarity = 0.0

        for name, data in self.face_database.items():
            saved_features = np.array(data['features'])

            # 确保特征维度一致
            min_len = min(len(features), len(saved_features))
            if min_len > 0:
                # 计算余弦相似度
                similarity = sk_metrics.cosine_similarity(
                    [features[:min_len]],
                    [saved_features[:min_len]]
                )[0][0]

                if similarity > best_similarity and similarity > self.recognition_threshold:
                    best_similarity = similarity
                    best_match = name

        return best_match, best_similarity

    def recognize(self, image: np.ndarray) -> List[Tuple[str, Tuple, float]]:
        """识别图像中的人脸"""
        results = []

        try:
            # 检测人脸
            faces = self.detect_faces(image)

            for (x, y, w, h) in faces:
                # 提取人脸区域
                if y + h <= image.shape[0] and x + w <= image.shape[1]:
                    face_roi = image[y:y + h, x:x + w]

                    if face_roi.size > 0 and face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                        # 提取特征
                        features = self.extract_face_features(face_roi)

                        # 识别
                        name, confidence = self.recognize_face(features)

                        results.append((name, (x, y, w, h), confidence))

        except Exception as e:
            logging.error(f"人脸识别失败: {e}")

        return results

    def capture_face(self, image: np.ndarray, name: str) -> Tuple[bool, str]:
        """采集新人脸"""
        try:
            faces = self.detect_faces(image)

            if len(faces) == 0:
                return False, "未检测到人脸，请调整位置"
            elif len(faces) > 1:
                return False, "检测到多个人脸，请确保只有一个人脸在画面中"
            else:
                x, y, w, h = faces[0]
                face_roi = image[y:y + h, x:x + w]

                if face_roi.size > 0 and face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                    # 提取特征
                    features = self.extract_face_features(face_roi)

                    # 保存到数据库
                    self.face_database[name] = {
                        'features': features.tolist(),
                        'capture_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'samples': 1
                    }

                    self.save_face_database()
                    return True, f"成功采集 {name} 的人脸数据"
                else:
                    return False, "人脸区域无效"

        except Exception as e:
            logging.error(f"采集人脸失败: {e}")
            return False, f"采集失败: {str(e)}"

    def get_database_info(self) -> Dict:
        """获取数据库信息"""
        return {
            'total_faces': len(self.face_database),
            'names': list(self.face_database.keys()),
            'last_update': time.strftime("%Y-%m-%d %H:%M:%S")
        }