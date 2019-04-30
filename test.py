import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
def testing():
	slides = []
	ppt = []
	ans_range = []
	index = 0
	for folder in os.listdir('Dataset'):
		index_start = index
		for filename in os.listdir('Dataset/'+folder):
			img = cv.imread(os.path.join("Dataset",folder,filename))
			imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			edgy = edge_detection(imgray)
			img = crop_image(edgy)
			if filename == "ppt.jpg":
				ppt.append(img)
				ans_range.append((index_start,index))
			else:
				slides.append(img)
			index = index + 1
	slide_index = 0
	for img in slides:
		mx = 10.0
		idx = -1
		ppt_index = 0
		for possibility in ppt:
			cur_max = max(np.corrcoeff(img, possibility))
			if cur_max > mx:
				mx = cur_max
				idx = ppt_index
			ppt_index = ppt_index + 1
		if ans_range[idx][0] > slide_index or ans_range[idx][1] < slide_index:
			print("Error : ",img)
		slide_index = slide_index + 1
testing()
