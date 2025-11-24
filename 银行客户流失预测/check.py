import pandas as pd

# 替换为你的数据集路径
data = pd.read_csv(r"D:\机器学习\银行客户流失预测\supply_chain_train.csv")
print("数据集中的所有列名：")
print(data.columns.tolist())  # 打印所有列名