import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from pprint import *
from tqdm import tqdm


def edge_detection(im):
    dx = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=5)
    dy = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=5)
    gradient = np.sqrt(np.multiply(dx, dx) + np.multiply(dy, dy))
    max_val = np.amax(gradient)
    min_val = np.amin(gradient)
    norm_gradient = np.around(255 * (gradient - min_val) / (max_val - min_val))
    norm_gradient = norm_gradient.astype(np.uint8)
    return norm_gradient


def crop_image(im, w, h):
    colsum = np.sum(im, axis=0)
    rowsum = np.sum(im, axis=1)
    w_mid = w // 2
    h_mid = h // 2
    idx1 = np.argmax(colsum[6:w_mid])
    idx2 = np.argmax(colsum[w_mid:w - 6])
    idx3 = np.argmax(rowsum[6:h_mid])
    idx4 = np.argmax(rowsum[h_mid:h - 6])
    val1 = colsum[idx1]
    val2 = colsum[idx2]
    val3 = rowsum[idx3]
    val4 = rowsum[idx4]
    if val1 + val2 + val3 + val4 < 100000:
        return im
    else:
        return im[idx1:idx2 - idx1 + w_mid, idx3:idx4 - idx3 + h_mid]


def testing():
    slides = []
    slide_name = []
    ppt = []
    ppt_name = []
    ans_range = []
    index = 0
    for folder in os.listdir('Dataset'):
        index_start = index
        for filename in os.listdir('Dataset/' + folder):
            img = cv.imread(os.path.join("Dataset", folder, filename))
            # print(filename, folder)
            imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # print(img.shape, folder, filename)
            h, w, _ = img.shape
            edgy = edge_detection(imgray)
            if filename == 'ppt.jpg':
                ppt.append(imgray)
                ppt_name.append(folder+filename)
            else:
                img = crop_image(edgy, w, h)
                slide_name.append(folder+filename)
                slides.append(img)
                index = index + 1
        ans_range.append((index_start, index-1))
    slide_index = 0
    for img in tqdm(slides):
        mx = -10.0
        idx = -1
        ppt_index = 0
        for possibility in ppt:
            # print(img.shape, possibility.size)
            h, w = img.shape 
            cur_max = np.amax(np.corrcoef(img, cv.resize(possibility, (w, h))))
            if cur_max > mx:
                mx = cur_max
                idx = ppt_index
            ppt_index = ppt_index + 1
        if ans_range[idx][0] > slide_index or ans_range[idx][1] < slide_index:
            print("Error ",slide_name[slide_index], " ",ppt_name[idx])
        else:
            print(slide_name[slide_index])
        slide_index = slide_index + 1
testing()
