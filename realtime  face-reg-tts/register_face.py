# 导入必要的库
import os  # 操作系统接口，用于文件和目录操作
import cv2  # OpenCV库，用于图像处理和摄像头操作
import argparse  # 命令行参数解析库
from logger import logger  # 日志记录器
from face_database import face_db, FACE_DB_DIR  # 人脸数据库实例和数据库目录
from tts_engine import tts_engine  # 文本转语音引擎实例

def capture_face(name):
    """
    通过摄像头捕获人脸照片
    
    Args:
        name: 要注册的人脸对应的姓名
    
    Returns:
        str: 成功时返回保存的照片路径，失败时返回None
    """
    # 打开摄像头（默认使用第0个摄像头）
    video_capture = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not video_capture.isOpened():
        logger.error("无法打开摄像头")
        return None
    
    # 记录操作提示信息
    logger.info(f"请面对摄像头，按空格键拍照，按ESC退出")
    
    # 进入循环，持续显示摄像头画面并等待用户操作
    while True:
        # 读取摄像头的一帧图像
        ret, frame = video_capture.read()
        if not ret:  # 如果读取失败，继续尝试
            continue
        
        # 在画面上显示注册信息和操作提示
        cv2.putText(frame, f"正在注册: {name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 绿色文字
        cv2.putText(frame, "按空格键拍照，按ESC退出", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # 黄色文字
        
        # 显示实时摄像头画面
        cv2.imshow('人脸注册', frame)
        
        # 等待按键输入，参数1表示1毫秒后超时继续
        key = cv2.waitKey(1) & 0xFF
        
        # 如果用户按下空格键，拍摄照片
        if key == ord(' '):
            # 构建保存路径，使用姓名作为文件名
            image_path = os.path.join(FACE_DB_DIR, f"{name}.jpg")
            
            # 保存图像到文件
            cv2.imwrite(image_path, frame)
            logger.info(f"照片已保存到: {image_path}")
            
            # 释放摄像头资源和关闭窗口
            video_capture.release()
            cv2.destroyAllWindows()
            
            # 返回保存的照片路径
            return image_path
        
        # 如果用户按下ESC键，取消注册
        elif key == 27:
            logger.info("用户取消注册")
            # 释放资源
            video_capture.release()
            cv2.destroyAllWindows()
            return None

def register_face_from_camera(name):
    """
    通过摄像头注册人脸到数据库
    包含人脸拍摄、人脸特征提取和数据库存储功能
    
    Args:
        name: 要注册的人脸对应的姓名
    
    Returns:
        bool: 注册是否成功
    """
    try:
        # 调用capture_face函数通过摄像头拍摄人脸照片
        image_path = capture_face(name)
        if not image_path:  # 如果用户取消拍摄或拍摄失败
            return False
        
        # 将拍摄的照片和姓名添加到人脸数据库
        success = face_db.add_face(image_path, name)
        
        # 根据添加结果进行不同处理
        if success:
            # 注册成功
            logger.info(f"人脸注册成功: {name}")
            tts_engine.speak(f"{name} 人脸注册成功")  # 语音提示注册成功
        else:
            # 注册失败，清理拍摄的照片
            if os.path.exists(image_path):
                os.remove(image_path)
            logger.error(f"人脸注册失败: {name}")
            tts_engine.speak(f"{name} 人脸注册失败，请重试")  # 语音提示注册失败
        
        return success
    except Exception as e:
        # 捕获所有异常，确保程序不会崩溃
        logger.error(f"注册人脸时出错: {str(e)}")
        tts_engine.speak("注册过程中发生错误")  # 语音提示发生错误
        return False

def register_face_from_file(image_path, name):
    """
    通过图片文件注册人脸到数据库
    适用于批量注册或已有照片的情况
    
    Args:
        image_path: 包含人脸的图片文件路径
        name: 要注册的人脸对应的姓名
    
    Returns:
        bool: 注册是否成功
    """
    try:
        # 检查指定的图片文件是否存在
        if not os.path.exists(image_path):
            logger.error(f"文件不存在: {image_path}")
            return False
        
        # 从图片中提取人脸特征并添加到数据库
        success = face_db.add_face(image_path, name)
        
        # 根据添加结果进行不同处理
        if success:
            # 注册成功
            logger.info(f"从文件注册人脸成功: {name}")
            tts_engine.speak(f"{name} 人脸注册成功")  # 语音提示注册成功
            
            # 复制原始图片到人脸数据库目录，保持数据库文件的一致性
            target_path = os.path.join(FACE_DB_DIR, f"{name}.jpg")
            import shutil
            shutil.copy2(image_path, target_path)  # copy2保留文件元数据
            logger.info(f"照片已复制到人脸数据库: {target_path}")
        else:
            # 注册失败
            logger.error(f"从文件注册人脸失败: {name}")
            tts_engine.speak(f"{name} 人脸注册失败，请检查照片质量")  # 语音提示注册失败
        
        return success
    except Exception as e:
        # 捕获所有异常
        logger.error(f"从文件注册人脸时出错: {str(e)}")
        tts_engine.speak("注册过程中发生错误")  # 语音提示发生错误
        return False

def main():
    """
    程序主函数
    处理命令行参数并调用相应的人脸注册函数
    支持两种注册方式：
    1. 通过摄像头实时拍摄照片注册
    2. 通过已有的图片文件注册
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='人脸注册工具')
    
    # 添加必需的姓名参数
    parser.add_argument('--name', required=True, help='要注册的人名')
    
    # 添加可选的文件路径参数，不提供时默认使用摄像头
    parser.add_argument('--file', help='人脸照片文件路径，如果不提供则使用摄像头')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        # 记录开始注册的日志
        logger.info(f"开始注册人脸: {args.name}")
        
        # 语音提示开始注册
        tts_engine.speak(f"开始注册 {args.name} 的人脸")
        
        # 根据是否提供了文件路径，选择不同的注册方式
        if args.file:
            # 从指定的图片文件注册人脸
            register_face_from_file(args.file, args.name)
        else:
            # 通过摄像头拍摄照片并注册人脸
            register_face_from_camera(args.name)
            
    finally:
        # 无论注册成功与否，都确保TTS引擎停止，释放资源
        # 放在finally块中可以确保即使发生异常也能执行
        tts_engine.stop()


# 如果作为独立脚本运行，则调用main函数
# 这使得该脚本可以直接在命令行中运行，而不需要从其他模块导入
if __name__ == "__main__":
    main()