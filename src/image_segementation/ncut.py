import numpy as np


def ncut_graph_matrix(im, sigma_d=1e2, sigma_g=1e2):
    """用于创建归一化割的矩阵，其中sigma_d和sigma_g像素相似性的权重参数
       该函数获取图像数组，并利用输入的彩色图像RGB值或灰度图像的灰度值创建一个特征向量
    """

    m, n = im.shape[:2]
    N = m * n

    # 归一化，并创建RGB或灰度向量
    if len(im.shape) == 3:
        for i in range(3):
            im[:, :, i] = im[:, :, i].max()
        vim = im.reshape((-1, 3))  # 自动计算行数，列数为3
    else:
        im /= im.max()
        vim = im.flatten()  # 折叠为一维

    # x,y坐标用于距离计算
    xx, yy = np.meshgrid(range(n), range(m))
    x, y = xx.flatten(), yy.flatten()

    # 创建边线权重矩阵
    W = np.zeros((N, N), 'f')
    for i in range(N):
        for j in range(i, N):
            d = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            W[i, j] = W[j, i] = np.exp(-1.0 * sum((vim[i] - vim[j]) ** 2 / sigma_g)) * np.exp(-d / sigma_d)
    return W


from scipy.cluster.vq import *


def cluster(S, k, ndim):
    """从相似性矩阵进行谱聚类"""

    # 检查对称性
    if np.sum(np.abs(S - S.T)) > 1e-10:
        print("not symmetric")

    # 创建拉普拉斯矩阵
    rowsum = np.sum(np.abs(S), axis=0)
    D = np.diag(1 / np.sqrt(rowsum + 1e-6))
    L = np.dot(D, np.dot(S, D))

    # 计算L的特征向量
    U, sigma, V = np.linalg.svd(L)

    # 从前ndim个特征向量创建特征向量，
    # 堆叠向量特征作为矩阵的列
    features = np.array(V[:ndim]).T

    # K-means聚类
    features = whiten(features)
    centroids, distortion = kmeans(features, k)
    code, distance = vq(features, centroids)

    return code, V
