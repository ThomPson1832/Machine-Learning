import pandas as pd
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 查看数据集的前几行
print(df.head())
# 将目标变量转换为类别名称
df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])
print(df.head())
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图形风格
sns.set(style="whitegrid")

# 使用pairplot展示特征之间的关系
sns.pairplot(df, hue='target_name', markers=["o", "s", "D"])
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 将数据分为特征和目标变量
X = df.drop(['target', 'target_name'], axis=1)
y = df['target_name']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化SVM分类器
svc = SVC(kernel='linear')  # 这里我们使用线性核函数

# 训练模型
svc.fit(X_train, y_train)

# 进行预测
y_pred = svc.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 打印混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# 打印分类报告
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print('Classification Report:')
print(class_report)

# 假设我们有一个新的样本
new_sample = [[5.0, 3.6, 1.4, 0.2]]

# 进行预测
prediction = svc.predict(new_sample)
predicted_class = iris.target_names[prediction[0]]
print(f'Predicted class for new sample: {predicted_class}')
