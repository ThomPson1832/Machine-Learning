# 导入必要的库
from datetime import datetime  # 日期时间库，用于处理时间间隔和时间戳
from config import MIN_REPEAT_INTERVAL, MAX_ANNOUNCE_COUNT, ANNOUNCE_DELAY  # 导入配置参数
from logger import logger  # 日志记录器，用于记录系统运行信息
from tts_engine import tts_engine  # 文本转语音引擎，用于执行播报
import threading  # 线程库，用于实现异步播报
from collections import deque  # 双端队列，用于存储最近检测记录


class AnnouncementManager:
    """
    公告管理器类
    负责管理语音播报的策略，实现智能去重和多人识别时的智能播报
    支持批量播报和防止短时间内重复播报同一个人
    """
    
    def __init__(self):
        """初始化公告管理器，设置数据结构和状态"""
        # 记录每个人最后一次播报的时间戳，用于实现智能去重
        self.last_announced = {}
        # 线程锁，保证并发访问时的数据一致性
        self.lock = threading.Lock()
        # 存储最近的检测记录，用于统计分析
        self.recent_detections = deque(maxlen=50)  # 限制最多存储50条记录
        # 标记是否正在进行播报，防止重叠播报
        self.announcement_in_progress = False
        # 当前批次的人员集合
        self.current_batch = set()
    
    def should_announce(self, name):
        """
        判断是否应该播报某个人名
        实现智能去重，防止短时间内重复播报同一个人
        
        Args:
            name: 要检查的人名
            
        Returns:
            bool: True表示应该播报，False表示不应该播报
        """
        with self.lock:
            current_time = datetime.now()
            
            # 检查是否在最小重复间隔内
            if name in self.last_announced:
                last_time = self.last_announced[name]
                # 如果距离上次播报的时间小于配置的最小重复间隔，不进行播报
                if (current_time - last_time) < MIN_REPEAT_INTERVAL:
                    logger.debug(f"{name} 在重复间隔内，跳过播报")
                    return False
            
            return True  # 可以进行播报
    
    def update_last_announced(self, name):
        """
        更新最后播报时间
        记录某个人名最后一次被播报的时间，用于后续智能去重判断
        
        Args:
            name: 需要更新时间的人名
        """
        with self.lock:
            self.last_announced[name] = datetime.now()
    
    def process_recognition_result(self, recognized_faces):
        """
        处理人脸识别结果，应用智能播报策略
        1. 更新最近检测记录
        2. 根据去重规则筛选需要播报的人脸
        3. 启动异步播报线程
        
        Args:
            recognized_faces: 人脸识别结果列表，每个元素为(name, confidence)元组
        """
        if not recognized_faces:
            return  # 如果没有识别到人脸，直接返回
        
        # 更新最近检测记录，用于统计分析
        current_time = datetime.now()
        for name, confidence in recognized_faces:
            self.recent_detections.append((name, current_time))
        
        # 过滤出应该播报的人脸，应用智能去重规则
        to_announce = []
        with self.lock:
            for name, confidence in recognized_faces:
                if self.should_announce(name):
                    to_announce.append((name, confidence))
        
        # 如果有需要播报的人脸，并且当前没有播报在进行中，启动新的播报线程
        if to_announce and not self.announcement_in_progress:
            threading.Thread(target=self._announce_batch, args=(to_announce,), daemon=True).start()
    
    def _announce_batch(self, recognized_faces):
        """
        批量播报人脸识别结果
        实现多人识别时的智能播报策略
        
        Args:
            recognized_faces: 需要播报的人脸识别结果列表
        """
        self.announcement_in_progress = True  # 标记开始播报
        try:
            # 根据置信度排序，优先播报置信度高的
            recognized_faces.sort(key=lambda x: x[1], reverse=True)
            
            # 限制最大播报人数，避免一次播报过多人名
            faces_to_announce = recognized_faces[:MAX_ANNOUNCE_COUNT]
            
            # 生成播报文本，根据人数采用不同的播报策略
            announcement_texts = []
            if len(faces_to_announce) == 1:
                # 单人人脸识别的情况
                name = faces_to_announce[0][0]
                announcement_texts.append(f"欢迎，{name} 同学")
                self.update_last_announced(name)  # 更新播报时间
            else:
                # 多人人脸识别的情况
                names = [face[0] for face in faces_to_announce]
                
                # 更新所有播报人员的最后播报时间
                for name in names:
                    self.update_last_announced(name)
                
                # 根据人数生成不同格式的播报文本
                if len(names) == 2:
                    # 两个人的情况，直接列出
                    announcement_texts.append(f"欢迎，{names[0]} 和 {names[1]} 同学")
                else:
                    # 超过2人的情况，采用更加自然的中文表达
                    announcement_texts.append(f"欢迎，{names[0]}、{names[1]}")
                    # 剩余的人单独播报，提升自然度
                    for name in names[2:]:
                        announcement_texts.append(f"以及 {name} 同学")
            
            # 执行播报
            if announcement_texts:
                logger.info(f"开始播报: {', '.join(announcement_texts)}")
                # 使用批量播报功能，按指定间隔依次播报所有文本
                tts_engine.speak_batch(announcement_texts, ANNOUNCE_DELAY)
        except Exception as e:
            # 捕获所有异常，避免播报过程中的错误影响系统运行
            logger.error(f"批量播报失败: {str(e)}")
        finally:
            # 无论成功还是失败，都标记播报结束
            self.announcement_in_progress = False
    
    def get_recognition_stats(self):
        """
        获取识别统计信息
        统计最近5分钟内每个人的识别次数
        
        Returns:
            dict: 包含人名和对应识别次数的字典
        """
        with self.lock:
            # 统计最近5分钟内的识别次数
            current_time = datetime.now()
            recent_5min = []
            # 筛选出最近5分钟内的检测记录
            for name, timestamp in self.recent_detections:
                if (current_time - timestamp).total_seconds() <= 300:  # 5分钟 = 300秒
                    recent_5min.append(name)
            
            # 计算每个人的识别次数
            stats = {}
            for name in recent_5min:
                stats[name] = stats.get(name, 0) + 1
            
            return stats
    
    def reset_announcement_history(self):
        """
        重置播报历史
        清除所有播报记录，使所有人都可以立即被重新播报
        """
        with self.lock:
            self.last_announced.clear()  # 清除最后播报时间记录
            self.recent_detections.clear()  # 清除最近检测记录
            logger.info("播报历史已重置")  # 记录重置操作


# 创建全局公告管理器实例
# 系统中的其他模块可以直接导入并使用此实例
announcement_manager = AnnouncementManager()