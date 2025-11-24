import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -------------------------- 配置参数 --------------------------
# 输出目录（自动创建）
OUTPUT_DIR = "./handwritten_digits"
# 图片尺寸（MNIST标准：28x28）
IMG_SIZE = (28, 28)
# 手写风格配置（调整字体和位置，模拟真实手写）
FONT_PATH = "C:/Windows/Fonts/seguisym.ttf"  # Windows系统默认字体（手写风格）
FONT_SIZES = [18, 20, 19, 21, 18, 20, 19, 21, 18]  # 1-9数字的字体大小（模拟差异）
# 数字位置偏移（模拟手写位置不工整）
POSITION_OFFSETS = [
    (4, 2), (3, 3), (4, 3),  # 1,2,3
    (3, 2), (4, 3), (3, 3),  # 4,5,6
    (4, 2), (3, 3), (4, 3)   # 7,8,9
]


def create_handwritten_digit(digit, font_path, font_size, position_offset):
    """
    生成单个手写数字图片
    :param digit: 数字（1-9）
    :param font_path: 字体文件路径
    :param font_size: 字体大小
    :param position_offset: 位置偏移（x, y）
    :return: 28x28灰度图的numpy数组
    """
    # 1. 创建黑底图片（MNIST是黑底白字）
    img = Image.new("L", IMG_SIZE, 0)  # "L"表示灰度模式，0=黑色
    draw = ImageDraw.Draw(img)

    # 2. 加载字体（若指定字体失败，使用默认字体）
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default(size=font_size)

    # 3. 绘制数字（白色，模拟手写笔迹）
    draw.text(position_offset, str(digit), fill=255, font=font)  # 255=白色

    # 4. 转为numpy数组（符合MNIST格式）
    img_np = np.array(img, dtype=np.uint8)
    return img_np


def save_handwritten_digits():
    """生成1-9的手写数字图片并保存"""
    # 1. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录：{OUTPUT_DIR}")

    # 2. 循环生成1-9的数字图片
    for i in range(1, 10):
        # 获取当前数字的配置（索引从0开始）
        idx = i - 1
        font_size = FONT_SIZES[idx]
        pos_offset = POSITION_OFFSETS[idx]

        # 生成数字图片
        digit_np = create_handwritten_digit(i, FONT_PATH, font_size, pos_offset)

        # 保存为PNG图片（文件名：digit_1.png ~ digit_9.png）
        img = Image.fromarray(digit_np)
        save_path = os.path.join(OUTPUT_DIR, f"digit_{i}.png")
        img.save(save_path)

        print(f"已保存：{save_path}")

    print(f"\n所有手写数字图片已保存到 {OUTPUT_DIR} 目录")
    print("图片规格：28x28像素、灰度图、黑底白字（符合MNIST格式）")


if __name__ == "__main__":
    save_handwritten_digits()