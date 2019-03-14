from numpy import *
import random


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
        est_prob = array([gauss(m, v, points) for m, v in zip(self.mean, self.var)])

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


def split_data(data, rate=0.8):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    # 打乱数据
    random.shuffle(data)
    # 按比例获取测试数据
    test_samples = data[int(len(data) * rate):-1]
    test_data = array([sample[0] for sample in test_samples])
    test_labels = array([sample[1] for sample in test_samples])
    # 按比例提取测试数据，并根据类别信息分组
    train_samples = data[:int(len(data) * rate)]
    train_dict = {}
    for sample in train_samples:
        label = sample[1]
        if label in train_dict:
            train_dict[label].append(sample[0])
        else:
            train_labels.append(label)
            train_dict.update({label: [sample[0]]})
    for label in train_labels:
        train_data.append(array(train_dict[label]))
    return [train_data, train_labels], [test_data, test_labels]


if __name__ == "__main__":
    data = []
    labels = []
    with open("iris.data", 'r')as f:
        for line in f:
            line = line.strip()
            line = line.split(',')
            nums = array([float(num) for num in line[:-1]])
            cls = line[-1]
            data.append([nums, cls])

    # print(data)
    train, test = split_data(data)
    # print(test[0][:10])
    bc = BayesClassifier()
    bc.train(train[0], train[1])
    print(bc.classify(test[0])[0])
    print(test[1])
