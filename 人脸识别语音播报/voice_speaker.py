import pyttsx3
import threading
import time
import logging


class VoiceSpeaker:
    """语音播报器"""

    def __init__(self):
        self.engine = pyttsx3.init()
        self.is_speaking = False
        self.speak_lock = threading.Lock()

        # 初始化语音设置
        self.init_voice_settings()

    def init_voice_settings(self):
        """初始化语音设置"""
        try:
            # 获取所有可用的语音
            voices = self.engine.getProperty('voices')

            # 优先选择中文语音
            chinese_voice = None
            for voice in voices:
                if 'chinese' in voice.id.lower() or 'mandarin' in voice.id.lower():
                    chinese_voice = voice
                    break

            if chinese_voice:
                self.engine.setProperty('voice', chinese_voice.id)
                logging.info(f"已设置中文语音: {chinese_voice.id}")
            else:
                # 使用默认语音
                self.engine.setProperty('voice', voices[0].id)
                logging.warning(f"未找到中文语音，使用默认语音: {voices[0].id}")

            # 默认设置
            self.set_rate(150)
            self.set_volume(0.8)

            logging.info("TTS引擎初始化成功")

        except Exception as e:
            logging.error(f"初始化语音引擎失败: {e}")
            self.engine = None

    def set_volume(self, volume):
        """设置音量（0.0 - 1.0）"""
        if self.engine and 0.0 <= volume <= 1.0:
            self.engine.setProperty('volume', volume)

    def set_rate(self, rate):
        """设置语速（50 - 300）"""
        if self.engine and 50 <= rate <= 300:
            self.engine.setProperty('rate', rate)

    def speak(self, text):
        """ speak text (blocking) """
        if not self.engine:
            return

        with self.speak_lock:
            self.is_speaking = True
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logging.error(f"语音播报失败: {e}")
            finally:
                self.is_speaking = False

    def speak_async(self, text):
        """异步播报文本（非阻塞）"""
        if not self.engine:
            return

        # 如果正在播报，则忽略
        if self.is_speaking:
            return

        thread = threading.Thread(target=self.speak, args=(text,), daemon=True)
        thread.start()

    def speak_face_result(self, names, mode="完整模式", enable_unknown_alert=True):
        """根据识别结果进行语音播报"""
        if not names:
            return

        # 去重
        unique_names = list(set(names))

        if mode == "简洁模式":
            # 简洁模式：只播报已知人员
            known_names = [name for name in unique_names if name != "未知人员"]
            if known_names:
                text = f"检测到{len(known_names)}个已知人员：{', '.join(known_names)}"
                self.speak_async(text)

        elif mode == "完整模式":
            # 完整模式：播报所有人员
            known_names = [name for name in unique_names if name != "未知人员"]
            unknown_count = names.count("未知人员")

            text_parts = []
            if known_names:
                text_parts.append(f"检测到{len(known_names)}个已知人员：{', '.join(known_names)}")
            if enable_unknown_alert and unknown_count > 0:
                text_parts.append(f"检测到{unknown_count}个未知人员")

            if text_parts:
                self.speak_async("，".join(text_parts))

        elif mode == "安全模式":
            # 安全模式：重点播报未知人员
            unknown_count = names.count("未知人员")
            known_names = [name for name in unique_names if name != "未知人员"]

            text_parts = []
            if enable_unknown_alert and unknown_count > 0:
                text_parts.append(f"警告：检测到{unknown_count}个未知人员")
            if known_names:
                text_parts.append(f"已知人员：{', '.join(known_names)}")

            if text_parts:
                self.speak_async("，".join(text_parts))

    def stop(self):
        """停止语音播报"""
        if self.engine:
            self.engine.stop()
            self.is_speaking = False

    def is_busy(self):
        """检查是否正在播报"""
        return self.is_speaking