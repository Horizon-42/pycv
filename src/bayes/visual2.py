import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()  # 加载机器学习的下的iris数据集，先来认识一下iris数据集的一些操作，其实iris数据集就是一个字典集。下面注释的操作，可以帮助理解

# print(iris.keys())  # 打印iris索引，关键字

# n_sample, n_features = iris.data.shape

# print(iris.data.shape[0])  # 样本
# print(iris.data.shape[1])  # 4个特征
#
# print(n_sample, n_features)
#
# print(iris.data[0])
#
# print(iris.target.shape)
# print(iris.target)  # 三个种类，分别用0,1,2来表示
# print(iris.target_names)  # 三个种类的英文名称
# print("feature_names:", iris.feature_names)

iris_setosa = iris.data[:50]  # 第一种花的数据

iris_versicolor = iris.data[50:100]  # 第二种花的数据

iris_virginica = iris.data[100:150]  # 第三种花的数据

# print(iris_setosa)


iris_setosa = np.hsplit(iris_setosa, 4)  # 运用numpy.hsplit水平分割获取各特征集合，分割成四列
iris_versicolor = np.hsplit(iris_versicolor, 4)
iris_virginica = np.hsplit(iris_virginica, 4)

setosa = {'sepal_length': iris_setosa[0], 'sepal_width': iris_setosa[1], 'petal_length': iris_setosa[2],
          'petal_width': iris_setosa[3]}

versicolor = {'sepal_length': iris_versicolor[0], 'sepal_width': iris_versicolor[1], 'petal_length': iris_versicolor[2],
              'petal_width': iris_versicolor[3]}

virginica = {'sepal_length': iris_virginica[0], 'sepal_width': iris_virginica[1], 'petal_length': iris_virginica[2],
             'petal_width': iris_virginica[3]}

size = 5  # 散点的大小
setosa_color = 'b'  # 蓝色代表setosa
versicolor_color = 'g'  # 绿色代表versicolor
virginica_color = 'r'  # 红色代表virginica

sepal_width_ticks = np.arange(2, 5, step=0.5)  # sepal_length分度值和刻度范围
sepal_length_ticks = np.arange(4, 8, step=0.5)  # sepal_width分度值和刻度范围
petal_width_ticks = np.arange(0, 2.5, step=0.5)  # petal_width分度值和刻度范围
petal_length_ticks = np.arange(1, 7, step=1)  # petal_length分度值和刻度范围

ticks = [sepal_length_ticks, sepal_width_ticks, petal_length_ticks, petal_width_ticks]
label_text = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']

# print(ticks)

plt.figure(figsize=(11, 11))  # 设置画布大小
plt.suptitle("Iris Set (blue=setosa, green=versicolour, red=virginca) ", fontsize=18)

for i in range(0, 4):
    for j in range(0, 4):
        plt.subplot(4, 4, i * 4 + j + 1)  # 创建子画布

        if i == j:
            print(i * 4 + j + 1)  # 序列号

            plt.xticks([])
            plt.yticks([])
            plt.text(0.1, 0.4, label_text[i], size=18)

        else:
            plt.scatter(iris_setosa[j], iris_setosa[i], c=setosa_color, s=size)
            plt.scatter(iris_versicolor[j], iris_versicolor[i], c=versicolor_color, s=size)
            plt.scatter(iris_virginica[j], iris_virginica[i], c=virginica_color, s=size)
            # plt.xlabel(label_text[j])
            # plt.ylabel(label_text[i])
            plt.xticks(ticks[j])
            plt.yticks(ticks[i])

# plt.show()  //需要保存的时候不能show，即调用savefig()保存图片的时候，不能调用show()，这个我也不知道为什么这样。。。

plt.savefig('iris.png', format='png')  # 保存图片