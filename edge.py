import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from pprint import *
from tqdm import tqdm
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)
import numpy as np


def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj())
    
    image = fftconvolve(np.square(image), a1) - \
            np.square(fftconvolve(image, a1)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return np.absolute(out)
def fftconvolve(a,b):
    ma,na = a.shape
    mb,nb = b.shape
    return np.fft.ifft2(np.fft.fft2(a,[2*ma-1,2*na-1])*np.fft.fft2(b,[2*mb-1,2*nb-1]))
# def normxcorr2(b,a):
# 	c = conv2(a,np.flipud(np.fliplr(b)))
# 	a = conv2(a**2, np.ones(b.shape))
# 	b = sum(b.flatten()**2)
# 	c = c/np.sqrt(a*b)
# 	return abs(c)
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
    if val1 + val2 + val3 + val4 < 10000:
        return im
    else:
        return im[idx1+6:idx2 + h_mid, idx3+6:idx4 + w_mid]
img = cv.imread(os.path.join("Dataset/02_2/ppt.jpg"))
sl = cv.imread(os.path.join("Dataset/02_2/4.jpg"))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sl = cv.cvtColor(sl, cv.COLOR_BGR2GRAY)
x = edge_detection(img)
sl = edge_detection(sl)
# print(sl)
threshold = 60
# sl[sl < threshold] = 0
# sl[sl >= threshold] = 255
# plt.imshow(sl)
# plt.show()
h, w = x.shape
x = crop_image(sl, w, h)
h, w = img.shape
cur_max = np.amax(normxcorr2(img, cv.resize(img, (w, h))))
# cur_max = np.amax(np.corrcoef(img, cv.resize(img, (w, h))))
print(cur_max)
# plt.imshow(sl)
# plt.show()
# cur_max = np.amax(np.corrcoef(img, cv.resize(sl, (w, h))))
# print(np.corrcoef(img, cv.resize(sl, (w, h))))
# mse = (np.square(x - sl)).mean(axis=None)
# print(mse)
