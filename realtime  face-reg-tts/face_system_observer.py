from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np


class FaceSystemObserver(QObject):
    """
    人脸识别系统观察者类
    采用观察者设计模式，用于接收和转发人脸识别系统的事件
    
    主要职责：
    1. 从人脸识别系统接收各类事件通知
    2. 通过信号机制将事件转发给UI界面
    3. 提供标准化的事件处理接口
    """
    
    # 定义信号
    face_recognized_signal = pyqtSignal(str, float)  # 当识别到人脸时发射 (姓名, 置信度)
    frame_processed_signal = pyqtSignal(np.ndarray)  # 当帧处理完成时发射 (处理后的帧)
    system_status_changed_signal = pyqtSignal(bool)  # 当系统状态变化时发射 (是否运行中)
    error_occurred_signal = pyqtSignal(str)  # 当发生错误时发射 (错误信息)
    
    def __init__(self):
        """
        初始化观察者
        设置初始状态并准备接收事件
        """
        super().__init__()
        # 可以在这里添加额外的初始化逻辑
        
    def on_face_recognized(self, name, confidence):
        """
        当识别到人脸时调用的回调方法
        
        参数:
            name: 识别到的人脸名称
            confidence: 识别置信度
        """
        # 转发识别结果信号给UI
        self.face_recognized_signal.emit(name, confidence)
    
    def on_frame_processed(self, frame):
        """
        当帧处理完成时调用的回调方法
        
        参数:
            frame: 处理后的视频帧 (numpy数组)
        """
        # 转发处理后的帧信号给UI
        self.frame_processed_signal.emit(frame)
    
    def on_system_status_changed(self, is_running):
        """
        当系统状态变化时调用的回调方法
        
        参数:
            is_running: 系统是否正在运行
        """
        # 转发系统状态变化信号给UI
        self.system_status_changed_signal.emit(is_running)
    
    def on_error(self, error_message):
        """
        当发生错误时调用的回调方法
        
        参数:
            error_message: 错误信息描述
        """
        # 转发错误信号给UI
        self.error_occurred_signal.emit(error_message)