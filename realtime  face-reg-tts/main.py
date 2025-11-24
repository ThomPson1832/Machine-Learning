# 导入必要的模块
import signal  # 用于处理系统信号（如Ctrl+C）
import sys     # 提供系统相关功能
from logger import logger  # 导入日志记录器
from tts_engine import tts_engine  # 导入语音合成引擎实例
from face_recognition_system import face_system  # 导入人脸识别系统实例
from face_database import face_db  # 导入人脸数据库实例
from config import config  # 导入配置实例
from PyQt5.QtWidgets import QApplication  # 导入PyQt5应用类
from main_window import MainWindow  # 导入主窗口类
import time    # 提供时间相关功能


def signal_handler(sig, frame):
    """
    信号处理函数，用于捕获并处理系统信号
    确保程序能够优雅退出，释放所有资源
    
    Args:
        sig: 信号编号
        frame: 当前堆栈帧
    """
    logger.info("接收到退出信号，正在停止系统...")
    # 停止人脸识别系统，释放摄像头和窗口资源
    face_system.stop()
    # 停止TTS引擎，释放语音资源
    tts_engine.stop()
    logger.info("系统已完全停止")
    sys.exit(0)  # 正常退出程序


def main():
    """
    主程序入口函数
    负责初始化系统组件并启动PyQt5用户界面
    """
    try:
        # 设置信号处理器，捕获用户中断信号(SIGINT)和终止信号(SIGTERM)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 打印系统启动信息
        logger.info("===============================")
        logger.info("实时人脸识别与语音播报系统启动中")
        logger.info("===============================")
        
        # 初始化TTS引擎并进行测试播报
        logger.info("正在初始化TTS引擎...")
        tts_engine.speak("人脸识别系统启动中，请稍候")
        time.sleep(2)  # 等待语音播报完成
        
        # 初始化PyQt5应用
        app = QApplication(sys.argv)
        app.setApplicationName("实时人脸识别与语音播报系统")
        app.setApplicationVersion("1.0.0")
        
        # 创建并显示主窗口
        logger.info("正在创建用户界面...")
        main_window = MainWindow(config, face_db, tts_engine, face_system)
        main_window.show()
        
        # 播报系统已启动信息
        tts_engine.speak("人脸识别系统已就绪")
        
        # 运行应用程序主循环
        exit_code = app.exec_()
        return exit_code
        
    except KeyboardInterrupt:
        # 处理用户主动中断的情况
        logger.info("用户中断程序")
    except Exception as e:
        # 处理所有其他异常
        logger.error(f"系统运行出错: {str(e)}", exc_info=True)  # 记录详细的错误信息和堆栈
        # 发生异常时通过语音通知用户
        tts_engine.speak("系统发生错误，请检查日志")
    finally:
        # 无论程序如何退出，都确保释放所有资源
        face_system.stop()
        tts_engine.stop()
        face_db.save_encodings()  # 保存人脸数据库
        logger.info("程序已退出")
        return 1


if __name__ == "__main__":
    # 确保只有当直接运行此文件时才执行main函数
    # 避免作为模块导入时自动执行
    main()