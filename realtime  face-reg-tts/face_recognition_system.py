# 导入必要的库
import cv2  # OpenCV库，用于视频捕获和图像处理
import face_recognition  # 人脸识别库，提供人脸检测和识别功能
import numpy as np  # 数值计算库，用于数组操作和数学计算
from logger import logger  # 导入日志记录器
from config import CAMERA_ID, RECOGNITION_TOLERANCE, FRAME_PROCESS_INTERVAL, MAX_RETRY_COUNT  # 导入配置参数
from face_database import face_db  # 导入人脸数据库实例
from announcement_manager import announcement_manager  # 导入公告管理器实例
import threading  # 多线程库，用于并发处理
import time  # 时间库，用于延时和计时


class FaceRecognitionSystem:
    """
    人脸识别系统主类
    负责摄像头管理、人脸检测、人脸识别、结果显示和异常处理
    使用多线程实现视频处理和显示的分离
    """
    
    def __init__(self):
        """初始化人脸识别系统，设置初始状态"""
        self.video_capture = None  # 视频捕获对象
        self.running = False  # 系统运行状态标志
        self.process_thread = None  # 视频处理线程
        self.frame_count = 0  # 帧计数器，用于控制处理频率
        self.last_exception_time = 0  # 上次异常发生时间
        self.retry_count = 0  # 异常重试计数
    
    def start(self):
        """启动人脸识别系统的主方法"""
        try:
            # 初始化摄像头，使用配置中指定的摄像头ID
            self.video_capture = cv2.VideoCapture(CAMERA_ID)
            # 检查摄像头是否成功打开
            if not self.video_capture.isOpened():
                raise Exception(f"无法打开摄像头，ID: {CAMERA_ID}")
            
            logger.info(f"摄像头启动成功，ID: {CAMERA_ID}")
            
            # 从目录初始化人脸数据库，加载已注册的人脸信息
            face_db.initialize_database_from_directory()
            
            # 设置运行标志并启动视频处理线程
            self.running = True
            # 创建守护线程，确保主线程结束时该线程也会结束
            self.process_thread = threading.Thread(target=self._process_frames, daemon=True)
            self.process_thread.start()
            
            logger.info("人脸识别系统已启动")
            
            # 启动显示窗口的主循环
            self._display_frames()
            
        except Exception as e:
            # 捕获并记录启动过程中的任何异常
            logger.error(f"启动人脸识别系统失败: {str(e)}")
            # 确保在异常情况下也能正确停止系统
            self.stop()
            raise  # 重新抛出异常，让上层处理
    
    def _process_frames(self):
        """
        视频帧处理线程函数
        在独立线程中执行人脸检测和识别，避免影响UI响应
        """
        while self.running:
            try:
                # 从摄像头读取一帧视频
                ret, frame = self.video_capture.read()
                # 检查是否成功读取帧
                if not ret:
                    logger.error("无法读取视频帧")
                    self._handle_exception()  # 处理读取失败异常
                    continue
                
                # 增加帧计数
                self.frame_count += 1
                
                # 实现降采样处理：每隔FRAME_PROCESS_INTERVAL帧处理一次
                # 这样可以显著降低CPU占用率
                if self.frame_count % FRAME_PROCESS_INTERVAL != 0:
                    time.sleep(0.01)  # 短暂休眠，让出CPU时间片
                    continue
                
                # 获取人脸数据库中的所有已注册人脸
                all_faces = face_db.get_all_faces()
                # 如果没有已注册的人脸，暂时跳过识别
                if not all_faces:
                    time.sleep(0.1)
                    continue
                
                # 解包人脸编码和对应的名称
                known_face_encodings = [encoding for _, encoding in all_faces]
                known_face_names = [name for name, _ in all_faces]
                
                # 将BGR格式转换为RGB格式
                # 注意：OpenCV使用BGR格式，而face_recognition库使用RGB格式
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 检测图像中的所有人脸位置
                # face_locations是一个包含人脸边界框坐标的列表
                face_locations = face_recognition.face_locations(rgb_frame)
                # 提取每个人脸的特征编码
                # face_encodings返回128维的人脸特征向量
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # 存储成功识别的人脸信息
                recognized_faces = []
                
                # 遍历检测到的每一个人脸
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # 计算当前人脸编码与数据库中所有人脸编码的距离
                    # 距离越小，相似度越高
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    
                    # 找到距离最小的匹配项（最相似的人脸）
                    best_match_index = np.argmin(face_distances)
                    best_match_distance = face_distances[best_match_index]
                    
                    # 根据预设的阈值判断是否为已知人脸
                    if best_match_distance < RECOGNITION_TOLERANCE:
                        # 匹配成功，获取对应的人名
                        name = known_face_names[best_match_index]
                        # 将距离转换为置信度（距离越小，置信度越高）
                        confidence = 1.0 - best_match_distance
                        recognized_faces.append((name, confidence))  # 保存识别结果
                        logger.info(f"识别到人脸: {name}, 置信度: {confidence:.2f}")
                    else:
                        # 匹配失败，标记为未知人脸
                        name = "未知"
                        logger.info(f"检测到未知人脸")
                    
                    # 在视频帧上绘制人脸框和标签
                    self._draw_face_box(frame, (top, right, bottom, left), name, best_match_distance)
                
                # 将识别结果传递给公告管理器，进行智能播报
                if recognized_faces:
                    announcement_manager.process_recognition_result(recognized_faces)
                
                # 成功处理一帧后，重置异常重试计数
                self.retry_count = 0
                
                time.sleep(0.01)  # 短暂休眠，避免CPU占用过高
                
            except Exception as e:
                # 捕获并记录处理过程中的异常
                logger.error(f"处理帧时出错: {str(e)}")
                self._handle_exception()  # 调用异常处理方法
                time.sleep(0.1)  # 发生异常后暂停一段时间再重试
    
    def _display_frames(self):
        """
        显示视频帧的主循环
        在主线程中执行，负责显示处理后的视频和处理用户输入
        """
        try:
            while self.running:
                # 读取当前帧用于显示
                ret, frame = self.video_capture.read()
                if not ret:
                    continue
                
                # 计算并显示FPS（每秒帧数）
                fps = self._calculate_fps()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 设置并显示系统状态
                status = "运行中"
                status_color = (0, 255, 0)  # 绿色表示正常运行
                
                # 显示已注册的人脸数量
                face_count = len(face_db.face_names)
                cv2.putText(frame, f"已注册: {face_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # 显示状态信息
                cv2.putText(frame, status, (frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                # 显示处理后的视频帧
                cv2.imshow('人脸识别系统', frame)
                
                # 等待用户按键，设置1毫秒延迟以保持界面响应
                # 按'q'键退出系统
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break
                
        except Exception as e:
            # 捕获并记录显示过程中的异常
            logger.error(f"显示帧时出错: {str(e)}")
        finally:
            # 无论如何都要确保停止系统，释放资源
            self.stop()
    
    def _draw_face_box(self, frame, face_location, name, distance):
        """
        在视频帧上绘制人脸框和标签
        
        Args:
            frame: 视频帧
            face_location: 人脸位置坐标 (top, right, bottom, left)
            name: 识别出的人名或"未知"
            distance: 匹配距离
        """
        # 解包人脸位置坐标
        top, right, bottom, left = face_location
        
        # 根据识别结果设置不同的颜色
        if name != "未知":
            color = (0, 255, 0)  # 绿色表示识别成功
        else:
            color = (0, 0, 255)  # 红色表示未知人脸
        
        # 绘制人脸矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # 生成标签文本，包含人名和置信度
        label = f"{name} ({1-distance:.2f})" if name != "未知" else name
        label_height = 30  # 标签背景高度
        
        # 绘制标签背景矩形
        cv2.rectangle(frame, (left, bottom - label_height), (right, bottom), color, cv2.FILLED)
        
        # 在标签背景上绘制文本
        cv2.putText(frame, label, (left + 6, bottom - 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _calculate_fps(self):
        """
        计算每秒帧数(FPS)
        返回估算的FPS值，实际应用中可以使用更精确的计时方法
        """
        # 这里使用简化的FPS计算方法
        # FRAME_PROCESS_INTERVAL是处理帧的间隔，乘以一个经验系数得到FPS
        return FRAME_PROCESS_INTERVAL * 10  # 估算值
    
    def _handle_exception(self):
        """
        异常处理方法，实现智能重试机制
        当连续发生异常时，尝试重新初始化摄像头
        """
        current_time = time.time()
        
        # 如果距离上次异常超过5秒，重置重试计数
        # 这是为了区分连续异常和间歇性异常
        if current_time - self.last_exception_time > 5:
            self.retry_count = 0
        
        # 增加重试计数
        self.retry_count += 1
        # 更新上次异常时间
        self.last_exception_time = current_time
        
        # 当重试次数达到最大限制时，尝试重新初始化摄像头
        if self.retry_count >= MAX_RETRY_COUNT:
            logger.warning("重试次数过多，尝试重新初始化摄像头")
            try:
                # 释放现有摄像头资源
                if self.video_capture:
                    self.video_capture.release()
                # 重新打开摄像头
                self.video_capture = cv2.VideoCapture(CAMERA_ID)
                # 重置重试计数
                self.retry_count = 0
                logger.info("摄像头重新初始化成功")
            except Exception as e:
                logger.error(f"重新初始化摄像头失败: {str(e)}")
    
    def stop(self):
        """
        停止人脸识别系统
        安全释放所有资源，包括摄像头、线程和窗口
        """
        try:
            # 设置运行标志为False，通知各个线程退出循环
            self.running = False
            
            # 等待处理线程结束，设置超时时间避免无限等待
            if self.process_thread:
                self.process_thread.join(timeout=2.0)
            
            # 释放摄像头资源
            if self.video_capture:
                self.video_capture.release()
            
            # 关闭所有OpenCV创建的窗口
            cv2.destroyAllWindows()
            
            logger.info("人脸识别系统已停止")
        except Exception as e:
            # 即使在停止过程中发生异常也要记录
            logger.error(f"停止系统时出错: {str(e)}")


# 创建全局人脸识别系统实例
# 这样其他模块可以直接导入并使用此实例
face_system = FaceRecognitionSystem()