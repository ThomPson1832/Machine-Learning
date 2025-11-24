# 导入必要的库
import os  # 操作系统接口，用于文件路径操作
import sys  # Python解释器相关功能
from logger import logger  # 日志记录模块，用于记录系统日志
from tts_engine import tts_engine  # 文本转语音引擎实例
from face_database import face_db, FACE_DB_DIR  # 人脸数据库实例和数据库目录
import time  # 时间模块，用于延时操作


def test_tts_engine():
    """
    测试文本转语音(TTS)引擎是否正常工作
    验证语音合成和播放功能是否正常
    
    Returns:
        bool: 测试是否成功通过
    """
    logger.info("测试TTS引擎...")
    try:
        # 调用TTS引擎播放测试语音
        tts_engine.speak("TTS引擎测试，这是一段中文语音测试")
        # 等待3秒，确保语音播放完成
        time.sleep(3)  # 等待语音播放完成
        logger.info("TTS引擎测试通过")
        return True
    except Exception as e:
        # 捕获并记录测试过程中的任何异常
        logger.error(f"TTS引擎测试失败: {str(e)}")
        return False


def test_face_database():
    """
    测试人脸数据库功能
    验证数据库目录是否存在以及保存功能是否正常
    
    Returns:
        bool: 测试是否成功通过
    """
    logger.info("测试人脸数据库...")
    try:
        # 检查人脸数据库目录是否存在
        if not os.path.exists(FACE_DB_DIR):
            logger.error("人脸数据库目录不存在")
            return False
        
        # 测试保存人脸编码功能
        face_db.save_encodings()
        # 记录当前已注册的人脸数量
        logger.info(f"当前注册人脸数量: {len(face_db.face_names)}")
        logger.info("人脸数据库测试通过")
        return True
    except Exception as e:
        # 捕获并记录测试过程中的任何异常
        logger.error(f"人脸数据库测试失败: {str(e)}")
        return False


def create_test_face():
    """
    创建一个测试用的人脸记录（模拟）
    生成一个假的人脸编码并添加到数据库中
    
    Returns:
        bool: 测试人脸是否成功创建和添加
    """
    logger.info("创建测试人脸记录...")
    try:
        # 创建一个假的编码（实际应用中应该是真实的人脸编码）
        import numpy as np
        # 创建一个128维的零向量作为测试用的人脸编码
        # 注：face_recognition库使用128维向量表示人脸特征
        test_encoding = np.zeros(128)  # face_recognition使用128维编码
        
        # 检查测试用户是否已存在，不存在则添加
        if "测试用户" not in face_db.face_names:
            face_db.face_encodings.append(test_encoding)  # 添加人脸编码
            face_db.face_names.append("测试用户")  # 添加对应的用户名
            face_db.save_encodings()  # 保存到数据库
            logger.info("测试人脸已添加")
        else:
            logger.info("测试人脸已存在")
        
        return True
    except Exception as e:
        # 捕获并记录测试过程中的任何异常
        logger.error(f"创建测试人脸失败: {str(e)}")
        return False


def main():
    """
    运行所有系统测试
    依次执行TTS引擎测试、人脸数据库测试和创建测试人脸测试
    
    Returns:
        bool: 所有测试是否都成功通过
    """
    logger.info("开始系统测试...")
    
    # 定义测试项目列表，包含测试名称和对应的测试函数
    tests = [
        ("TTS引擎", test_tts_engine),
        ("人脸数据库", test_face_database),
        ("创建测试人脸", create_test_face)
    ]
    
    # 统计成功通过的测试数量
    success_count = 0
    
    # 执行每个测试项目
    for name, test_func in tests:
        logger.info(f"\n--- 测试: {name} ---")
        # 如果测试通过，增加成功计数
        if test_func():
            success_count += 1
    
    # 打印总体测试结果
    logger.info(f"\n测试完成: {success_count}/{len(tests)} 个测试通过")
    
    # 根据测试结果进行总结和播报
    if success_count == len(tests):
        # 所有测试通过
        logger.info("所有测试通过！系统准备就绪")
        tts_engine.speak("系统测试全部通过，欢迎使用实时人脸识别系统")
    else:
        # 部分测试失败
        logger.warning("部分测试失败，请检查系统配置")
        tts_engine.speak("系统测试发现问题，请查看日志")
    
    # 等待语音播放完成
    time.sleep(3)
    
    # 停止TTS引擎，释放资源
    tts_engine.stop()
    
    # 返回测试是否全部通过
    return success_count == len(tests)


# 当脚本作为主程序运行时执行测试
if __name__ == "__main__":
    # 执行主测试函数并获取结果
    success = main()
    # 根据测试结果设置程序退出码
    # 0表示成功，1表示失败
    sys.exit(0 if success else 1)