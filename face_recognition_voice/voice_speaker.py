# voice_speaker.py - 增强版语音播报模块
import pyttsx3
import threading
import time
import logging
from typing import Optional, Dict, List


class VoiceSpeaker:
    """智能语音播报器"""

    def __init__(self):
        self.engine = None
        self.is_speaking = False
        self.speech_queue = []
        self.current_thread = None
        self.voice_settings = {
            'volume': 0.8,
            'rate': 150,
            'voice_id': None
        }
        self.initialize_engine()

        # 播报模式配置
        self.speak_modes = {
            "简洁模式": "{name}",
            "完整模式": "发现{name}",
            "安全模式": "发现{name}",
            "静音模式": None
        }

        # 播报历史记录
        self.speak_history = []
        self.max_history = 100

        logging.info("语音播报器初始化完成")

    def initialize_engine(self):
        """初始化语音引擎"""
        try:
            self.engine = pyttsx3.init()

            # 设置默认参数
            self.engine.setProperty('rate', self.voice_settings['rate'])
            self.engine.setProperty('volume', self.voice_settings['volume'])

            # 尝试设置中文语音
            self.set_chinese_voice()

            # 设置事件回调
            self.engine.connect('started-utterance', self.on_speech_start)
            self.engine.connect('finished-utterance', self.on_speech_end)

        except Exception as e:
            logging.error(f"语音引擎初始化失败: {e}")
            self.engine = None

    def set_chinese_voice(self):
        """设置中文语音"""
        if not self.engine:
            return

        try:
            voices = self.engine.getProperty('voices')
            chinese_voices = []

            for voice in voices:
                voice_info = {
                    'id': voice.id,
                    'name': voice.name,
                    'languages': getattr(voice, 'languages', []),
                    'gender': getattr(voice, 'gender', 'unknown')
                }

                # 检查是否为中文语音
                if any('zh' in str(lang).lower() or 'chinese' in str(lang).lower()
                       or '中文' in voice.name.lower() for lang in voice_info['languages']):
                    chinese_voices.append(voice_info)

            if chinese_voices:
                # 优先选择女性语音
                female_voices = [v for v in chinese_voices if 'female' in v['gender'].lower() or '女' in v['name']]
                if female_voices:
                    selected_voice = female_voices[0]
                else:
                    selected_voice = chinese_voices[0]

                self.engine.setProperty('voice', selected_voice['id'])
                self.voice_settings['voice_id'] = selected_voice['id']
                logging.info(f"已设置中文语音: {selected_voice['name']}")
            else:
                logging.warning("未找到中文语音，使用默认语音")

        except Exception as e:
            logging.error(f"设置中文语音失败: {e}")

    def set_volume(self, volume: float):
        """设置音量 0.0-1.0"""
        self.voice_settings['volume'] = max(0.0, min(1.0, volume))
        if self.engine:
            self.engine.setProperty('volume', self.voice_settings['volume'])

    def set_rate(self, rate: int):
        """设置语速"""
        self.voice_settings['rate'] = max(50, min(300, rate))
        if self.engine:
            self.engine.setProperty('rate', self.voice_settings['rate'])

    def set_speak_mode(self, mode: str):
        """设置播报模式"""
        return self.speak_modes.get(mode, "发现{name}")

    def speak(self, text: str, async_mode: bool = True):
        """语音播报"""
        if not self.engine or not text:
            return

        def _speak():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logging.error(f"语音播报失败: {e}")

        if async_mode:
            # 异步播报，不阻塞主线程
            thread = threading.Thread(target=_speak, daemon=True)
            thread.start()
        else:
            _speak()

    def speak_face_result(self, names: List[str], mode: str, enable_unknown_alert: bool = False):
        """智能人脸识别结果播报"""
        if not names or mode == "静音模式":
            return

        # 过滤未知人员（根据配置）
        known_names = [name for name in names if name != "未知人员"]
        unknown_names = [name for name in names if name == "未知人员"]

        # 构建播报文本
        speak_text = ""
        if known_names:
            name_list = "、".join(known_names)
            template = self.speak_modes.get(mode, "发现{name}")
            speak_text = template.format(name=name_list)

        # 安全模式下的未知人员警报
        if enable_unknown_alert and unknown_names:
            alert_text = "发现未知人员，请注意安全！"
            if speak_text:
                speak_text += "。" + alert_text
            else:
                speak_text = alert_text

        if speak_text:
            self.speak(speak_text)
            # 记录播报历史
            self.record_speech(speak_text)

    def record_speech(self, text: str):
        """记录播报历史"""
        record = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'text': text
        }
        self.speak_history.append(record)

        # 限制历史记录数量
        if len(self.speak_history) > self.max_history:
            self.speak_history.pop(0)

    def get_speech_history(self, limit: int = 10) -> List[Dict]:
        """获取播报历史"""
        return self.speak_history[-limit:]

    def on_speech_start(self, name):
        """语音开始回调"""
        self.is_speaking = True
        logging.info(f"开始播报: {name}")

    def on_speech_end(self, name, completed):
        """语音结束回调"""
        self.is_speaking = False
        if completed:
            logging.info(f"播报完成: {name}")
        else:
            logging.warning(f"播报中断: {name}")

    def stop(self):
        """停止语音引擎"""
        if self.engine:
            try:
                self.engine.stop()
                self.engine = None
            except:
                pass