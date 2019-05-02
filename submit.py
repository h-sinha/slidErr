import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from pprint import *
from tqdm import tqdm
import math
import sys

def edge_detection(im):
    im = cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    im = cv.GaussianBlur( im, (3,3), 0);
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
    threshold = np.double(0.02)
    if val1 + val2 + val3 + val4 < threshold*np.double(np.sum(colsum)):
        return im
    else:
        return im[idx1+6:idx2 + h_mid, idx3+6:idx4 + w_mid]
def testing():
    ppt_directory = sys.argv[1]
    frame_directory = sys.argv[2]
    slides = []
    slide_name = []
    ppt = []
    ppt_name = []
    for file in os.listdir(ppt_directory):
        img = cv.imread(os.path.join(ppt_directory, file))
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h, w, _ = img.shape
        img = edge_detection(imgray)
        ppt.append(img)
        ppt_name.append(file)
    for file in os.listdir(frame_directory):
        img = cv.imread(os.path.join(frame_directory, file))
        imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h, w, _ = img.shape
        img = edge_detection(imgray)
        slides.append(img)
        slide_name.append(file)
    slide_index = 0
    for img in slides:
        mx = -100000.0
        idx = -1
        ppt_index = 0
        for possibility in ppt:
            h, w = img.shape 
            img_mean = img.mean()
            img_dev = img.std()
            img_normalized = [(x - img_mean)/img_dev for x in img]
            resized_possibility = cv.resize(possibility, (w, h))
            resized_mean = resized_possibility.mean()
            resized_dev = resized_possibility.std()
            resized_normalized = [(x - resized_mean)/resized_dev for x in resized_possibility]
            numerator = np.sum(np.multiply(img_normalized, resized_possibility))
            denominator = np.sqrt(np.multiply(np.sum(np.square(img_normalized)), 
                np.sum(np.square(resized_possibility))))
            cur_max = numerator/denominator
            if cur_max > mx:
                mx = cur_max
                idx = ppt_index
            ppt_index = ppt_index + 1
        print(slide_name[slide_index]," ",ppt_name[idx])
        slide_index = slide_index + 1
testing()
