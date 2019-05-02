import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from pprint import *
from tqdm import tqdm
import math

def edge_detection(im):
    im = cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    dx = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=5)
    dy = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=5)
    gradient = np.sqrt(np.multiply(dx, dx) + np.multiply(dy, dy))
    max_val = np.amax(gradient)
    min_val = np.amin(gradient)
    norm_gradient = np.around(255 * (gradient - min_val) / (max_val - min_val))
    norm_gradient = norm_gradient.astype(np.uint8)
    # norm_gradient = cv.bilateralFilter(norm_gradient,9,75,75)
    # norm_gradient = cv.GaussianBlur( norm_gradient, (3,3), 0);
    # norm_gradient = cv.medianBlur(norm_gradient,5)
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
    # print(rowsum.size)
    # print(val1+val2+val3+val4, idx1, idx2, idx3, idx4)
    threshold = np.double(0.02)
    if val1 + val2 + val3 + val4 < threshold*np.double(np.sum(colsum)):
        return im
    else:
        return im[idx1+6:idx2 + h_mid, idx3+6:idx4 + w_mid]
def testing():
    slides = []
    slide_name = []
    ppt = []
    ppt_name = []
    ans_range = []
    index = 0
    img = cv.imread(os.path.join("Dataset", "12_9", "1.jpg"))
    h, w, _ = img.shape
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edgy = edge_detection(imgray)
    img = edgy
    # img = crop_image(edgy, w, h)
    img_mean = img.mean()
    img_dev = img.std()
    img_normalized = [(x - img_mean)/img_dev for x in img]
    plt.imshow(img_normalized)
    plt.show()
testing()
#09_322.jpg   08_1ppt.jpg
