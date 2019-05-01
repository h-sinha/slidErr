import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from pprint import *
from tqdm import tqdm
import math
# dst = cv2.filter2D(img,-1,kernel)
# dx = conv2([-1 0 1; -2 0 2; -1 0 1], imgray);
# dy = conv2([-1 -2 -1; 0 0 0; 1 2 1], imgray);
# gradient = sqrt((dx.*dx) + (dy.*dy));
# max_val = max(max(gradient));
# min_val = min(min(gradient));
# norm_gradient = uint8(round(255*(gradient-min_val)/(max_val-min_val)));
# end
def conv2(a,b):
    ma,na = a.shape
    mb,nb = b.shape
    return np.fft.ifft2(np.fft.fft2(a,[2*ma-1,2*na-1])*np.fft.fft2(b,[2*mb-1,2*nb-1]))
def normxcorr2(b,a):
    c = conv2(a,np.flipud(np.fliplr(b)))
    a = conv2(a**2, np.ones(b.shape))
    b = sum(b.flatten()**2)
    c = c/np.sqrt(a*b)
    return c

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
    for folder in tqdm(os.listdir('Dataset')):
        index_start = index
        for filename in os.listdir('Dataset/' + folder):
            img = cv.imread(os.path.join("Dataset", folder, filename))
            # print(filename, folder)
            imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # print(img.shape, folder, filename)
            h, w, _ = img.shape
            edgy = edge_detection(imgray)
            if filename == 'ppt.jpg':
                img = crop_image(edgy, w, h)
                ppt.append(img)
                ppt_name.append(folder+" "+filename)
            else:
                img = crop_image(edgy, w, h)
                slide_name.append(folder+" "+filename)
                slides.append(img)
                index = index + 1
        ans_range.append((index_start, index-1))
    slide_index = 0
    correct = 0
    wrong = 0
    # f = open("errors_v2.txt", "a+")
    # f.write("Slide name --- Ppt name predicted")
    for img in tqdm(slides):
        mx = -100000.0
        idx = -1
        ppt_index = 0
        for possibility in ppt:
            # print(img.shape, possibility.size)
            h, w = img.shape 
            # cur_max = (np.square(img - cv.resize(possibility, (w, h)))).mean(axis=None)
            # print((np.corrcoef(img, cv.resize(possibility, (w, h)))).dtype)
            # cur_max = (np.corrcoef(img, cv.resize(possibility, (w, h))))[1][0]
            # print(cur_max,mx)
            img_mean = img.mean()
            img_normalized = [x - img_mean for x in img]
            resized_possibility = cv.resize(possibility, (w, h))
            resized_mean = resized_possibility.mean()
            resized_normalized = [x - resized_mean for x in resized_possibility]
            numerator = np.sum(np.multiply(img_normalized, resized_possibility))
            denominator = np.sqrt(np.multiply(np.sum(np.square(img_normalized)), 
                np.sum(np.square(resized_possibility))))
            cur_max = numerator/denominator
            # print(numerator, denominator, cur_max)
            if cur_max > mx:
                mx = cur_max
                idx = ppt_index
            ppt_index = ppt_index + 1
            # print(cur_max)
        if ans_range[idx][0] > slide_index or ans_range[idx][1] < slide_index:
            # slide_index = slide_index + 1
            # continue
            print(ans_range[idx], slide_index)
            print("Error ",slide_name[slide_index], " ",ppt_name[idx]," ", cur_max)
            newpath = os.path.join("Error_v3", str(wrong))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            cv.imwrite(os.path.join(newpath , str(slide_name[slide_index])), slides[slide_index])
            cv.imwrite(os.path.join(newpath , str(ppt_name[idx])), ppt[idx])
            wrong = wrong + 1
            # f.write(str(slide_name[slide_index]) + " " + str(ppt_name[idx])+"\n")
        else:
            correct = correct + 1
            print(slide_name[slide_index])
        slide_index = slide_index + 1
    print("Correct : ", correct, "Wrong : ", wrong, "Total : ", correct+wrong)
testing()
#09_322.jpg   08_1ppt.jpg
