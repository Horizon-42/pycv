import numpy as np
# from PIL import Image

import ncut
from scipy.misc import imresize
import os
import cv2

from matplotlib import pyplot as plt

path = r'/home/ai/桌面/works/WrinklesDetect/res/microspur'
files = os.listdir(path)
for fname in files:
    print("loading img...")
    src = cv2.imread(os.path.join(path, fname))
    im = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    m, n = im.shape[:2]
    # im = im.convert('L')

    print("imresize...")
    wid = 100
    rim = imresize(im, (wid, wid), interp='bilinear')
    rim = np.array(rim, 'f')

    print("get graph_matrix")
    A = ncut.ncut_graph_matrix(rim, sigma_d=1, sigma_g=1e-2)

    print("cluster...")
    code, V = ncut.cluster(A, k=2, ndim=1)

    codeim = imresize(code.reshape(wid, wid), (m, n), interp='nearest')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(codeim)
    plt.subplot(1, 2, 2)
    plt.imshow(im)
    plt.gray()
    plt.show()
