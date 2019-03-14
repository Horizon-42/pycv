import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)

iris = pd.DataFrame(load_iris().data)
#               花萼长             花萼宽           花瓣长度          花瓣宽度
iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
iris['Species'] = load_iris().target

if __name__ == "__main__":
    # 输出数据集大体情况
    print(iris.shape)
    print(iris.head())
    print()
    # 检测是否为均衡分类
    print(iris['Species'].value_counts())

    # 根据花萼绘制分布趋势
    plt.figure(0)
    sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)
    plt.savefig("Sepal.jpg")
    # 根据花瓣绘制分布趋势
    plt.figure(1)
    sns.jointplot(x="PetalLengthCm", y="PetalWidthCm", data=iris, size=5)
    plt.savefig("Petal.jpg")

    # 根据类别上色
    plt.figure(2)
    sns.FacetGrid(iris, hue="Species", size=5).map(plt.scatter, 'SepalLengthCm', 'SepalWidthCm')
    plt.savefig("SepalHue.jpg")
    plt.figure(3)
    sns.FacetGrid(iris, hue="Species", size=5).map(plt.scatter, 'PetalLengthCm', 'PetalWidthCm')
    plt.savefig("PetalHue.jpg")
