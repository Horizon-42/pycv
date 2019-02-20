from numpy import *
from numpy import random
from scipy.ndimage import filters
import rof
from PIL import *
import os
from matplotlib import pyplot as plt

# im = zeros((500, 500))
# im[100:400, 100:400] = 128
# im[200:300, 200:300] = 255
# im = im + 30 * random.standard_normal((500, 500))

path = r'/home/ai/桌面/works/WrinklesDetect/res/microspur'
files = os.listdir(path)
for fname in files:
    im = Image.open(os.path.join(path, fname))
    im = array(im.convert('L'))

    U, T = rof.denoise(im, im)
    G = filters.gaussian_filter(im, 10)

    from scipy.misc import imsave

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(G)
    plt.subplot(1, 3, 2)
    plt.imshow(U)
    plt.subplot(1, 3, 3)
    plt.imshow(T)
    plt.gray()
    plt.show()
