# 导入必要的库
import os  # 操作系统接口，用于文件和目录操作
import face_recognition  # 人脸识别库，用于生成和处理人脸编码
from logger import logger  # 日志记录器，用于记录操作日志
from config import FACE_DB_DIR  # 导入人脸数据库目录配置
import pickle  # 序列化模块，用于保存和加载人脸数据


class FaceDatabase:
    """
    人脸数据库类
    负责管理人脸数据的存储、加载、添加、删除和更新
    使用pickle进行数据序列化，支持从目录批量导入人脸数据
    """
    
    def __init__(self):
        """
        初始化人脸数据库
        创建两个空列表用于存储人脸编码和对应的人名
        设置编码文件路径并加载已保存的编码数据
        """
        # 存储人脸编码的列表，每个编码是128维的向量
        self.face_encodings = []
        # 存储人脸名称的列表，与face_encodings一一对应
        self.face_names = []
        # 设置人脸编码文件路径，使用pickle格式保存
        self.encoding_file = os.path.join(FACE_DB_DIR, 'face_encodings.pkl')
        # 初始化时自动加载已保存的编码数据
        self.load_encodings()
    
    def load_encodings(self):
        """
        从文件加载人脸编码数据
        检查编码文件是否存在，若存在则反序列化数据
        分别加载人脸编码和对应的人名到内存中
        """
        try:
            if os.path.exists(self.encoding_file):
                # 打开并读取pickle文件
                with open(self.encoding_file, 'rb') as f:
                    data = pickle.load(f)
                    # 加载人脸编码，若不存在则返回空列表
                    self.face_encodings = data.get('encodings', [])
                    # 加载人脸名称，若不存在则返回空列表
                    self.face_names = data.get('names', [])
                # 记录成功加载的人脸数量
                logger.info(f"已加载 {len(self.face_names)} 个人脸数据")
            else:
                # 文件不存在时记录信息
                logger.info("人脸数据库文件不存在，创建新的数据库")
        except Exception as e:
            # 捕获所有异常并记录错误
            logger.error(f"加载人脸数据库失败: {str(e)}")
    
    def save_encodings(self):
        """
        保存人脸编码数据到文件
        将内存中的人脸编码和人名序列化为字典格式
        使用pickle将数据保存到指定文件
        """
        try:
            # 组织要保存的数据结构
            data = {
                'encodings': self.face_encodings,  # 人脸编码列表
                'names': self.face_names  # 对应的人名列表
            }
            # 序列化并保存到文件
            with open(self.encoding_file, 'wb') as f:
                pickle.dump(data, f)
            # 记录成功保存的人脸数量
            logger.info(f"已保存 {len(self.face_names)} 个人脸数据")
        except Exception as e:
            # 捕获所有异常并记录错误
            logger.error(f"保存人脸数据库失败: {str(e)}")
    
    def add_face(self, image_path, name):
        """
        添加人脸到数据库
        从指定图像中检测人脸并提取编码，然后添加到数据库
        
        Args:
            image_path: 包含人脸的图像文件路径
            name: 人脸对应的人名
            
        Returns:
            bool: 添加成功返回True，失败返回False
        """
        try:
            # 加载图像文件
            image = face_recognition.load_image_file(image_path)
            
            # 使用HOG模型检测图像中的人脸位置
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                # 未检测到人脸时记录警告
                logger.warning(f"在图像 {image_path} 中未检测到人脸")
                return False
            
            # 根据检测到的人脸位置提取人脸编码
            face_encodings = face_recognition.face_encodings(image, face_locations)
            if not face_encodings:
                # 无法提取编码时记录警告
                logger.warning(f"无法提取 {image_path} 中的人脸编码")
                return False
            
            # 添加第一张人脸的编码和对应的人名到数据库
            self.face_encodings.append(face_encodings[0])
            self.face_names.append(name)
            
            # 保存更新后的数据库
            self.save_encodings()
            
            # 记录成功添加信息
            logger.info(f"成功添加人脸: {name}")
            return True
        except Exception as e:
            # 捕获所有异常并记录错误
            logger.error(f"添加人脸失败: {str(e)}")
            return False
    
    def get_all_faces(self):
        """
        获取所有已注册的人脸数据
        
        Returns:
            list: 包含(name, encoding)元组的列表，每个元组对应一个人脸
        """
        # 将人名和对应的人脸编码一一配对，转换为列表返回
        return list(zip(self.face_names, self.face_encodings))
    
    def delete_face(self, name):
        """
        删除指定名称的人脸数据
        查找并删除指定人名及其对应的人脸编码
        
        Args:
            name: 要删除的人脸名称
            
        Returns:
            bool: 删除成功返回True，失败或未找到返回False
        """
        try:
            if name in self.face_names:
                # 查找人名对应的索引位置
                index = self.face_names.index(name)
                # 删除对应的人名和人脸编码
                del self.face_names[index]
                del self.face_encodings[index]
                # 保存更新后的数据库
                self.save_encodings()
                # 记录成功删除信息
                logger.info(f"已删除人脸: {name}")
                return True
            else:
                # 未找到指定人名时记录警告
                logger.warning(f"未找到名称为 {name} 的人脸数据")
                return False
        except Exception as e:
            # 捕获所有异常并记录错误
            logger.error(f"删除人脸失败: {str(e)}")
            return False
    
    def update_face(self, image_path, name):
        """
        更新指定名称的人脸数据
        通过先删除旧数据再添加新数据的方式实现更新
        
        Args:
            image_path: 包含新人脸的图像文件路径
            name: 要更新的人脸名称
            
        Returns:
            bool: 更新成功返回True，失败返回False
        """
        # 先删除旧数据
        self.delete_face(name)
        # 添加新数据
        return self.add_face(image_path, name)
    
    def initialize_database_from_directory(self):
        """
        从face_database目录初始化数据库
        扫描指定目录中的所有图像文件，文件名（不含扩展名）作为人名
        将找到的人脸图像批量添加到数据库
        """
        try:
            # 遍历目录中的所有文件
            for filename in os.listdir(FACE_DB_DIR):
                # 检查是否为图像文件且不是编码文件
                if filename.endswith(('.jpg', '.jpeg', '.png')) and filename != 'face_encodings.pkl':
                    # 构建完整图像路径
                    image_path = os.path.join(FACE_DB_DIR, filename)
                    # 从文件名提取人名（不包含扩展名）
                    name = os.path.splitext(filename)[0]
                    
                    # 检查是否已存在该人名，避免重复添加
                    if name not in self.face_names:
                        self.add_face(image_path, name)
                    else:
                        logger.info(f"人脸 {name} 已存在，跳过")
        except Exception as e:
            # 捕获所有异常并记录错误
            logger.error(f"从目录初始化数据库失败: {str(e)}")


# 创建全局人脸数据库实例
# 其他模块可以直接导入此实例使用，避免重复创建数据库对象
face_db = FaceDatabase()