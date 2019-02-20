# -*- coding:utf-8 -*-
import numpy as np
from numpy import *


def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """

    :param im:含有噪声的输入图像(灰度图像)
    :param U_init:U的初始值
    :param tolerance:TV正则项权值
    :param tau:步长
    :param tv_weight:停止条件
    :return:去噪和去除纹理后的图像
    :return:纹理残留
    """

    m, n = im.shape
    U = U_init
    Px = im  # 对偶域的x分量
    Py = im  # 对偶域的y分量
    error = 1

    # 梯度下降法？
    while (error > tolerance):
        Uold = U

        # 原始变量的梯度
        GradUx = roll(U, -1, axis=1) - U  # 滚动数组元素 变量U梯度的x分量
        GradUy = roll(U, -1, axis=0) - U  # 变量U梯度的y分量

        # 更新对偶变量
        PxNew = Px + (tau / tv_weight) * GradUx
        PyNew = Py + (tau / tv_weight) * GradUy
        NormNew = maximum(1, sqrt(PxNew ** 2 + PyNew ** 2))

        # 更新对偶分量
        Px = PxNew / NormNew
        Py = PyNew / NormNew

        # 更新原始变量
        RxPx = roll(Px, 1, axis=1)
        RyPy = roll(Py, 1, axis=0)

        DivP = (Px - RxPx) + (Py - RyPy)  # 对偶域的散度
        U = im + tv_weight * DivP

        error = np.linalg.norm(U - Uold) / np.sqrt(n * m)

    return U, im - U
