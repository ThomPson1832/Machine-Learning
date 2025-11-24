import os
import glob

data_dir = r'D:\机器学习\CNN军事装备识别\war_tech_by_GonTech'
total = 0
classes = os.listdir(data_dir)

print("数据集样本统计：")
print("=" * 30)

for cls in classes:
    path = os.path.join(data_dir, cls)
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
        count = len(files)
        total += count
        print(f"{cls}: {count}个样本")

print("=" * 30)
print(f"总样本数: {total}")