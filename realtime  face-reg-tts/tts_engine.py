# 导入必要的库
import pyttsx3  # 文字转语音库，用于中文语音合成
import threading  # 多线程库，用于语音播放的异步处理
import queue  # 队列模块，用于存储待播放的文本
from logger import logger  # 导入日志记录器
from config import TTS_RATE, TTS_VOLUME  # 导入TTS配置参数


class TTSEngine:
    """
    中文语音合成引擎类
    负责初始化TTS引擎、管理语音播放队列、提供语音播放接口
    实现异步语音播放，避免阻塞主线程
    """
    
    def __init__(self):
        """初始化TTS引擎对象"""
        self.engine = None  # pyttsx3引擎实例
        self.queue = queue.Queue()  # 语音播放队列
        self.is_running = False  # 运行状态标志
        self.thread = None  # 语音播放线程
        self._initialize_engine()  # 初始化引擎
        self._start_processing_thread()  # 启动处理线程
    
    def _initialize_engine(self):
        """
        初始化TTS引擎
        1. 创建pyttsx3引擎实例
        2. 设置中文语音（如果可用）
        3. 配置语速和音量
        """
        try:
            # 创建pyttsx3引擎实例
            self.engine = pyttsx3.init()
            # 设置中文语音
            voices = self.engine.getProperty('voices')
            
            # 尝试选择中文语音（Windows系统下）
            for voice in voices:
                # 检查是否为中文语音，通过id中的关键词判断
                if 'chinese' in voice.id.lower() or 'china' in voice.id.lower() or 'zh' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    logger.info(f"已设置中文语音: {voice.id}")
                    break
            else:
                logger.warning("未找到中文语音包，使用默认语音")
            
            # 根据配置设置语速和音量
            self.engine.setProperty('rate', TTS_RATE)  # 语速
            self.engine.setProperty('volume', TTS_VOLUME)  # 音量
            
            logger.info(f"TTS引擎初始化成功，语速: {TTS_RATE}，音量: {TTS_VOLUME}")
        except Exception as e:
            logger.error(f"TTS引擎初始化失败: {str(e)}")
            self.engine = None
    
    def _start_processing_thread(self):
        """
        启动语音处理线程
        创建守护线程，确保主线程结束时该线程也会结束
        """
        self.is_running = True  # 设置运行状态为True
        # 创建守护线程来处理语音队列
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()  # 启动线程
        logger.info("语音处理线程已启动")
    
    def _process_queue(self):
        """
        处理语音队列
        持续从队列中获取文本并播放，支持异步语音播放
        通过None值作为退出信号
        """
        while self.is_running:
            try:
                # 从队列获取文本，阻塞等待
                text = self.queue.get()
                if text is None:  # 退出信号
                    break
                
                # 播报语音
                if self.engine:
                    self.engine.say(text)  # 将文本添加到TTS引擎的播放队列
                    self.engine.runAndWait()  # 执行语音播放，直到完成
                
                # 标记任务完成，确保队列计数正确
                self.queue.task_done()
                logger.debug(f"语音播报完成: {text}")
            except Exception as e:
                logger.error(f"处理语音队列时出错: {str(e)}")
                # 即使出错也要标记任务完成，避免死锁
                try:
                    self.queue.task_done()
                except:
                    pass
    
    def speak(self, text):
        """
        添加文本到语音队列进行播放
        
        Args:
            text: 要播放的文本字符串
        """
        if not text or not self.is_running:
            return
        
        try:
            self.queue.put(text)  # 将文本添加到队列
            logger.info(f"已添加语音播报: {text}")
        except Exception as e:
            logger.error(f"添加语音播报失败: {str(e)}")
    
    def speak_batch(self, texts, delay=0):
        """
        批量添加文本到语音队列
        
        Args:
            texts: 文本字符串列表
            delay: 文本之间的延迟时间（秒），默认为0
        """
        for i, text in enumerate(texts):
            self.speak(text)  # 播放当前文本
            # 如果设置了延迟，除了最后一个都添加延迟
            if i < len(texts) - 1 and delay > 0:
                import time
                time.sleep(delay)  # 等待指定的延迟时间
    
    def stop(self):
        """
        停止TTS引擎
        1. 设置运行状态为False
        2. 发送退出信号到队列
        3. 等待线程结束
        4. 停止引擎并释放资源
        """
        try:
            self.is_running = False  # 设置运行状态为False
            # 发送退出信号到队列，使处理线程能够退出
            self.queue.put(None)
            
            # 等待线程结束，设置超时防止死锁
            if self.thread:
                self.thread.join(timeout=5.0)
            
            # 停止引擎并释放资源
            if self.engine:
                self.engine.stop()  # 停止当前播放
                self.engine = None  # 释放引擎实例
            
            logger.info("TTS引擎已停止")
        except Exception as e:
            logger.error(f"停止TTS引擎时出错: {str(e)}")
    
    def clear_queue(self):
        """
        清空语音队列
        移除所有等待播放的文本，避免队列积压
        """
        try:
            # 循环清空队列中的所有元素
            while not self.queue.empty():
                self.queue.get_nowait()  # 非阻塞方式获取队列元素
                self.queue.task_done()  # 标记任务完成
            logger.info("语音队列已清空")
        except queue.Empty:
            pass  # 队列为空时无需处理
        except Exception as e:
            logger.error(f"清空语音队列时出错: {str(e)}")

# 创建全局TTS引擎实例
# 系统中的其他模块可以直接导入并使用此实例
# 这样可以确保整个系统使用同一个TTS引擎实例
# 避免资源浪费和可能的冲突
tts_engine = TTSEngine()