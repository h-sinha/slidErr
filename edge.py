import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from pprint import *
from tqdm import tqdm
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

def edge_detection(im):
    # dx = cv.filter2D(im, 2, np.matrix('-1 0 1; -2 0 2; -1 0 1'))
    # dy = cv.filter2D(im, 2, np.matrix('-1 -2 -1; 0 0 0; 1 2 1'))
    dx = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=5)
    dy = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=5)
    gradient = np.sqrt(np.multiply(dx, dx) + np.multiply(dy, dy))
    max_val = np.amax(gradient)
    min_val = np.amin(gradient)
    norm_gradient = np.around(255 * (gradient - min_val) / (max_val - min_val))
    norm_gradient = norm_gradient.astype(np.uint8)
    return norm_gradient
def crop_image(im, w, h):
    colsum = np.sum(im, axis=1)
    rowsum = np.sum(im, axis=0)
    w_mid = w // 2
    h_mid = h // 2
    idx1 = np.argmax(colsum[6:h_mid])
    idx2 = np.argmax(colsum[h_mid:h - 6])
    idx3 = np.argmax(rowsum[6:w_mid])
    idx4 = np.argmax(rowsum[w_mid:w - 6])
    val1 = colsum[idx1+6]
    val2 = colsum[idx2+h_mid]
    val3 = rowsum[6+idx3]
    val4 = rowsum[idx4+w_mid]
    print(rowsum.size)
    print(val1+val2+val3+val4, idx1, idx2, idx3, idx4)
    if val1 + val2 + val3 + val4 < 10000:
        return im
    else:
        return im[idx1+6:idx2 + h_mid, idx3+6:idx4 + w_mid]
img = cv.imread(os.path.join("Dataset/02_2/ppt.jpg"))
sl = cv.imread(os.path.join("Dataset/02_1/1.jpg"))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sl = cv.cvtColor(sl, cv.COLOR_BGR2GRAY)
x = edge_detection(img)
sl = edge_detection(sl)
# plt.imshow(sl)
# plt.show()
h, w = sl.shape
sl = crop_image(sl, w, h)
plt.imshow(sl)
plt.show()
# cur_max = np.amax(np.corrcoef(img, cv.resize(sl, (w, h))))
# print(np.corrcoef(img, cv.resize(sl, (w, h))))
# mse = (np.square(x - sl)).mean(axis=None)
# print(mse)
