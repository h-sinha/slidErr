import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from pprint import *
from tqdm import tqdm
import math

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

def testing():
    slides = []
    slide_name = []
    ppt = []
    ppt_name = []
    ans_range = []
    index = 0
    for folder in tqdm(os.listdir('Dataset')):
        index_start = index
        for filename in os.listdir('Dataset/' + folder):
            img = cv.imread(os.path.join("Dataset", folder, filename))
            imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            h, w, _ = img.shape
            edgy = edge_detection(imgray)
            if filename == 'ppt.jpg':
                img = edgy
                ppt.append(img)
                ppt_name.append(folder+" "+filename)
            else:
                img = edgy
                slide_name.append(folder+" "+filename)
                slides.append(img)
                index = index + 1
        ans_range.append((index_start, index-1))
    slide_index = 0
    correct = 0
    wrong = 0
    for img in tqdm(slides):
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
        if ans_range[idx][0] > slide_index or ans_range[idx][1] < slide_index:
            print(ans_range[idx], slide_index)
            print("Error ",slide_name[slide_index], " ",ppt_name[idx]," ", cur_max)
            newpath = os.path.join("Error_v6", str(wrong))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            cv.imwrite(os.path.join(newpath , str(slide_name[slide_index])), slides[slide_index])
            cv.imwrite(os.path.join(newpath , str(ppt_name[idx])), ppt[idx])
            wrong = wrong + 1
        else:
            correct = correct + 1
            print(slide_name[slide_index])
        slide_index = slide_index + 1
    print("Correct : ", correct, "Wrong : ", wrong, "Total : ", correct+wrong)
testing()
