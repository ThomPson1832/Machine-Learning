# download_haar_cascade.py - 下载Haar级联文件
import urllib.request
import os


def download_haar_cascade():
    """下载Haar级联分类器文件"""
    print("正在下载Haar级联分类器文件...")

    # Haar级联文件URL
    haar_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

    try:
        # 下载文件
        urllib.request.urlretrieve(haar_url, "haarcascade_frontalface_default.xml")
        print("✅ 已下载 haarcascade_frontalface_default.xml")

        # 检查文件是否下载成功
        if os.path.exists("haarcascade_frontalface_default.xml"):
            file_size = os.path.getsize("haarcascade_frontalface_default.xml")
            print(f"✅ 文件下载成功，大小: {file_size} 字节")
        else:
            print("❌ 文件下载失败")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("将使用备用的人脸检测方法")


if __name__ == "__main__":
    download_haar_cascade()