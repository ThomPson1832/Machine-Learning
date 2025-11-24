import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 加载数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 标签数据

# 转换为DataFrame以便查看
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])
print("数据集前5行（原始特征）：")
print(df[iris.feature_names + ['target']].head())
print("\n数据集前5行（含类别名称）：")
print(df[iris.feature_names + ['target', 'target_name']].head())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练SVM模型
model = SVC()
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
# 修正：使用内置的round()函数，而不是调用浮点数的round()方法
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 2))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 预测新数据
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
new_data_df = pd.DataFrame(new_data, columns=iris.feature_names)
prediction = model.predict(new_data_df)

if len(prediction) > 0:
    predicted_class_index = prediction[0]
    if 0 <= predicted_class_index < len(iris.target_names):
        predicted_class = iris.target_names[predicted_class_index]
        print(f"\n预测结果：{predicted_class}")
    else:
        print(f"\n预测索引无效：{predicted_class_index}")
else:
    print("\n未得到有效预测结果")
