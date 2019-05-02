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
def match():
    try:
        ppt_directory = sys.argv[1]
        frame_directory = sys.argv[2]
    except:
        print("Usage : python3 20171010.py <path/to/slides/directory> <path/to/frames/directory>")
        return
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
    f = open("20171010.txt","w+")
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
        f.write(slide_name[slide_index]+" "+ppt_name[idx]+"\n")
        slide_index = slide_index + 1
    f.close()
match()
