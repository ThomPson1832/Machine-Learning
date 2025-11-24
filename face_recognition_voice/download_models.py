# download_models.py - 下载人脸检测模型
import urllib.request
import os


def download_models():
    print("正在下载人脸检测模型...")

    # 模型文件URL
    proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"

    # 下载文件
    try:
        urllib.request.urlretrieve(proto_url, "deploy.prototxt")
        print("✅ 已下载 deploy.prototxt")

        urllib.request.urlretrieve(model_url, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        print("✅ 已下载模型文件")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("将使用备用的人脸检测方法")


if __name__ == "__main__":
    download_models()