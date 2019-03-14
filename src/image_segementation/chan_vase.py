from image_denoise import rof
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import cv2 as cv

path = r'/home/ai/桌面/works/WrinklesDetect/res/microspur'
files = os.listdir(path)
for fname in files:
    # im = np.array(Image.open(os.path.join(path, fname)).convert('L'))
    src = cv.imread(os.path.join(path, fname))
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    [h, s, v] = cv.split(hsv)
    im = s
    U, T = rof.denoise(im, im, tolerance=0.1)
    t = 0.4
    D = U < t * U.max()

    plt.figure()
    plt.gray()
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.subplot(1, 3, 2)
    plt.imshow(U)
    plt.subplot(1, 3, 3)
    plt.imshow(D)

    plt.show()
