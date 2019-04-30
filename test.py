import cv2 as cv
import os
import matplotlib.pyplot as plt
def testing():
	slides = []
	ppt = []
	ans_range = []
	index = 0
	for folder in os.listdir('slidErr/Dataset'):
		index_start = index
		for filename in os.listdir('slidErr/Dataset/'+folder):
			img = cv.imread(os.path.join("slidErr/Dataset",folder,filename))
			imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			edgy = edge_detection(imgray)
			img = crop_image(edgy)
			if filename == "ppt.jpg":
				ppt.append(img)
				ans_range.append((index_start,index))
			else:
				slides.append(img)
			index = index + 1
	for img in slides:
		# find correlation and match	
testing()
