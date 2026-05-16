from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils._repr_html import estimator


# 可视化鸢尾花数据
def visualize_data():
    # 1. 加载sklearn 内置的鸢尾花数据集
    iris_date = load_iris()

    # print(iris_date)
    # print(f"数据集: {iris_date.data}")
    # print(f"特征名称: {iris_date.feature_names}")
    # print(f"标签名称:{iris_date.target_names}")
    # print(f"标签: {iris_date.target}")

    iris_d = pd.DataFrame(iris_date.data, columns=iris_date.feature_names)
    iris_d['species'] = iris_date.target

    col1 = 'sepal length (cm)'
    col2 = 'sepal width (cm)'

    # 3. sns.lmpot()显示
    sns.lmplot(x=col1, y=col2, data=iris_d, hue='species', fit_reg=False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('iris data')
    plt.tight_layout()
    plt.show()

# 切分数据集
def split_data():
    iris_data = load_iris()
    x_train,x_test,y_train,y_test =train_test_split(iris_data.data,iris_data.target,test_size =0.3, random_state = 1)

    print(f"数据总数量:{len(iris_data.data)}")
    print(f"训练集中的x-特征值:{len(x_train)}")
    print(f"测试集中的x-标签值:{len(x_test)}")
    print(y_train)

# 模型的训练和预测
# 1.获取数据值
# 2.数据的基本处理
# 3.数据几预处理,数据标准化 归一化
# 4.模型训练
# 5.模型评估
# 6.模型的预测
def train_and_predict():
    iris_data = load_iris()
    x_train,x_test,y_train,y_test =train_test_split(iris_data.data,iris_data.target,test_size =0.3, random_state = 1)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    estimator = KNeighborsClassifier(n_neighbors=5)
    estimator.fit(x_train,y_train)
    print(f"准确率:{estimator.score(x_test,y_test)}")

# 交叉验证和网格搜索
def cross_validate_and_grid_search():
    # 1.数据的准备
    iris_data = load_iris()

    # 2.数据集的分割
    x_train,x_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target, test_size = 0.3,random_state = 1)

    # 3.数据的预处理 标准化
    scaler  = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 构建模型 来网格搜索
    estimator = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator = estimator,param_grid = {'n_neighbors':[i for i in range(1,11)]},cv = 5,scoring = 'accuracy')
    grid_search.fit(x_train,y_train)
    print(f"最佳参数:{grid_search.best_estimator_}")
    print(f"最佳结果:{grid_search.best_score_}")

    best_estimator = grid_search.best_estimator_
    test_arruracy = best_estimator.score(x_test,y_test)
    print(f"测试集准确率:{test_arruracy}")




if __name__ == '__main__':
    #  visualize_data()
    # split_data()
    #train_and_predict()
    cross_validate_and_grid_search()


