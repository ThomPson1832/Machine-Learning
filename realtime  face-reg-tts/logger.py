# 导入必要的库
import logging  # Python标准库中的日志模块
import os  # 操作系统接口，用于文件路径处理
from config import LOG_FILE, LOG_LEVEL  # 从配置文件导入日志相关设置


# 配置日志记录器
# 此函数负责设置和配置日志记录系统，包括创建日志目录、设置处理器和格式
# 返回配置好的logger实例，供全局使用
def setup_logger():
    # 创建logger实例
    # 使用名称'face_recognition_system'创建一个专门的日志记录器
    # 这样可以在大型应用中区分不同模块的日志
    logger = logging.getLogger('face_recognition_system')
    
    # 设置日志级别
    # 日志级别决定了哪些消息会被记录（DEBUG < INFO < WARNING < ERROR < CRITICAL）
    logger.setLevel(LOG_LEVEL)
    
    # 确保日志目录存在
    # 获取日志文件所在目录路径
    log_dir = os.path.dirname(LOG_FILE)
    # 如果目录不存在，则创建目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建文件处理器
    # 负责将日志写入到指定文件中，便于后续查看和分析
    # 指定使用UTF-8编码，确保中文等非ASCII字符能正确存储
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    # 设置文件处理器的日志级别
    file_handler.setLevel(LOG_LEVEL)
    
    # 创建控制台处理器
    # 负责将日志输出到控制台，便于开发和调试过程中实时监控
    console_handler = logging.StreamHandler()
    # 设置控制台处理器的日志级别
    console_handler.setLevel(LOG_LEVEL)
    
    # 定义日志格式
    # 格式包含时间戳、记录器名称、日志级别和消息内容
    # 这种格式便于阅读和后期分析日志
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 应用格式到文件处理器
    file_handler.setFormatter(formatter)
    # 应用格式到控制台处理器
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    # 检查是否已经有处理器，避免重复添加（例如在多次导入时）
    if not logger.handlers:
        logger.addHandler(file_handler)  # 添加文件处理器
        logger.addHandler(console_handler)  # 添加控制台处理器
    
    # 返回配置好的logger实例
    return logger


# 创建全局日志记录器
# 在模块级别创建全局logger实例，使其他模块可以直接导入使用
# 这样可以确保整个应用使用同一个日志配置
logger = setup_logger()

# 记录初始化日志
# 用于验证日志系统是否正常工作
logger.info("日志系统初始化成功")