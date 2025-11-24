# test_installation.py - 测试所有依赖是否正常
try:
    import cv2

    print("✅ OpenCV 安装成功")

    from PyQt5.QtWidgets import QApplication

    print("✅ PyQt5 安装成功")

    import pyttsx3

    print("✅ pyttsx3 安装成功")

    import numpy as np

    print("✅ numpy 安装成功")

    import sklearn

    print("✅ scikit-learn 安装成功")

    from PIL import Image

    print("✅ Pillow 安装成功")

    print("\n🎉 所有依赖安装成功！可以运行主程序了。")

except ImportError as e:
    print(f"❌ 依赖安装失败: {e}")