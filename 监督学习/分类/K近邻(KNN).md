# K近邻（KNN）

```python
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris  # 调取数据
from sklearn.model_selection import train_test_split  # 切分数据集为训练集和测试集
from sklearn.metrics import accuracy_score  # 计算分类预测的准确率

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['class'] = iris.target

target_map = {
    0: iris.target_names[0],
    1: iris.target_names[1],
    2: iris.target_names[2]
}

df['class'] = df['class'].map(target_map)

x = iris.data
y = iris.target.reshape(-1, 1)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3,  # test_size为测试集占比,其余为训练集
                                                    random_state=35,  # random_state为随机种子
                                                    stratify=y)  # stratify=y为按照y的分布做等比例分割

'''距离函数定义'''

def l1_distance(a, b):  # 曼哈顿距离
    return np.sum(np.abs(a - b), axis=1)

def l2_distance(a, b):  # 欧氏距离
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# 分类器实现
class kNN(object):
    # 定义一个初始化方法，__init__ 是类的构造方法
    def __init__(self, n_neighbors=1, dist_func=l1_distance):
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func

    # 训练模型方法
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    # 模型预测方法
    def predict(self, x):
        # 初始化预测分类数组
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)

        # 遍历输入的x数据点，取出每一个数据点的序号i和数据x_test
        for i, x_test in enumerate(x):
            # x_test跟所有训练数据计算距离
            distances = self.dist_func(self.x_train, x_test)

            # 得到的距离按照由近到远排序，取出索引值
            nn_index = np.argsort(distances)

            # 选取最近的k个点，保存它们对应的分类类别
            nn_y = self.y_train[nn_index[:self.n_neighbors]].ravel()  # .ravel()将二维矩阵转换为一维数组

            # 统计类别中出现频率最高的那个，赋给y_pred[i]
            y_pred[i] = np.argmax(np.bincount(nn_y))  # np.bincount()统计出现的次数

        return y_pred

'''测试'''

# 定义一个knn实例
knn = kNN(n_neighbors=3)
# 训练模型
knn.fit(x_train, y_train)
# 传入测试数据，做预测
y_pred = knn.predict(x_test)

print(y_test.ravel())
print(y_pred.ravel())

# 求出预测准确率
accuracy = accuracy_score(y_test, y_pred)

print("预测准确率: ", accuracy)

'''自动化对比'''

# 定义一个knn实例
knn = kNN()
# 训练模型
knn.fit(x_train, y_train)

# 保存结果list
result_list = []

# 针对不同的参数选取，做预测
for p in [1, 2]:
    knn.dist_func = l1_distance if p == 1 else l2_distance

    # 考虑不同的k取值，步长为2
    for k in range(1, 10, 2):
        knn.n_neighbors = k
        # 传入测试数据，做预测
        y_pred = knn.predict(x_test)
        # 求出预测准确率
        accuracy = accuracy_score(y_test, y_pred)
        result_list.append([k, 'l1_distance' if p == 1 else 'l2_distance', accuracy])
df = pd.DataFrame(result_list, columns=['k', '距离函数', '预测准确率'])
print(df)
```
