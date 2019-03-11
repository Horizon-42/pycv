from numpy import *


class BayesClassifier(object):
    def __init__(self):
        """使用训练数据初始化分类器"""
        self.labels = []
        self.mean = []
        self.var = []  # 类方差
        self.n = 0  # 类别数量

    def train(self, data, labels=None):
        """在数据data上训练，标记labels是可选的，默认为０...n-1"""
        if labels == None:
            labels = range(len(data))
        self.labels = labels
        self.n = len(labels)

        for c in data:
            self.mean.append(mean(c, axis=0))
            self.var.append(var(c, axis=0))

    def classify(self, points):
        """通过计算出的每一类的概率对数据点进行分类，并返回最可能的标记"""

        # 计算出每一类的概率
        est_prob = array(gauss(m, v, points) for m, v in zip(self.mean, self.var))

        # 获取具有最高概率的索引，该索引会给出类标签
        ndx = est_prob.argmax(axis=0)
        est_labels = array([self.labels[n] for n in ndx])

        return est_labels, est_prob


def gauss(m, v, x):
    """用独立均值ｍ和方差v评估ｄ维高斯分布"""

    if len(x.shape) == 1:
        n, d = 1, x.shape[0]
    else:
        n, d = x.shape

    S = diag(1 / v)
    x = x - m

    y = exp(-0.5 * diag(dot(x, dot(S, x.T))))

    return y * (2 * pi) ** (-d / 2.0) / (sqrt(prod(v)) + 1e-6)
