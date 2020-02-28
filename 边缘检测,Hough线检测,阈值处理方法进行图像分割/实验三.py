from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import cv2
import random
import math

#获取图片
def getimg():
  return Image.open("1.jpg")
  
#显示图片
def showimg(img, isgray=False):
  plt.axis("off")
  if isgray == True:
    plt.imshow(img, cmap='gray')
  else: 
    plt.imshow(img)
  plt.show()

def sobeldetect(img):
	imgshape = img.shape
	h = imgshape[0]
	w = imgshape[1]

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	dst = np.zeros((h,w,1),np.uint8)

	for i in range(0,h-2):
	    for j in range(0,w-2):
	        #计算x y方向的梯度
	        gy = gray[i,j]*1+gray[i,j+1]*2+gray[i,j+2]*1-gray[i+2,j]*1-gray[i+2,j+1]*2-gray[i+2,j+2]*1
	        gx = gray[i,j]*1+gray[i+1,j]*2+gray[i+2,j]*1-gray[i,j+2]*1-gray[i+1,j+2]*2-gray[i+2,j+2]*1
	        grad = math.sqrt(gx*gx+gy*gy)
	        if grad > 50:
	            dst[i,j] = 255
	        else:
	            dst[i,j] = 0
	return dst

def perwittdetect(img):
	imgshape = img.shape
	h = imgshape[0]
	w = imgshape[1]

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	dst = np.zeros((h,w,1),np.uint8)

	for i in range(0,h-2):
	    for j in range(0,w-2):
	        gy = gray[i,j]*1+gray[i,j+1]*1+gray[i,j+2]*1-gray[i+2,j]*1-gray[i+2,j+1]*1-gray[i+2,j+2]*1
	        gx = gray[i,j]*1+gray[i+1,j]*1+gray[i+2,j]*1-gray[i,j+2]*1-gray[i+1,j+2]*1-gray[i+2,j+2]*1
	        grad = math.sqrt(gx*gx+gy*gy)
	        if grad > 50:
	            dst[i,j] = 255
	        else:
	            dst[i,j] = 0
	return dst

def robertsdetect(img):
	imgshape = img.shape
	h = imgshape[0]
	w = imgshape[1]

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	dst = np.zeros((h,w,1),np.uint8)

	for i in range(0,h-1):
	    for j in range(0,w-1):
	        #计算x y方向的梯度
	        gy = gray[i,j]*1-gray[i+1,j+1]*1
	        gx = gray[i+1,j]*1-gray[i,j+1]*1
	        grad = math.sqrt(gx*gx+gy*gy)
	        if grad > 50:
	            dst[i,j] = 255
	        else:
	            dst[i,j] = 0
	return dst

def laplacedetect(img):
	imgshape = img.shape
	h = imgshape[0]
	w = imgshape[1]

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	new_image = np.zeros((h,w,1),np.uint8)
	L = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])     
	# L = np.array([[1,1,1],[1,-8,1],[1,1,1]])      
	for i in range(h-2):
	    for j in range(w-2):
	        new_image[i+1, j+1] = abs(np.sum(gray[i:i+3, j:j+3] * L))
	return new_image

def Hough_lines(img):

    height, width = img.shape[:2]
    accumulator = np.zeros((180, int(math.sqrt(height ** 2 + width ** 2))), dtype=np.int)

   
    lines = np.array([[0, 0], [0, 0]])

    line_length = 10

    for y in range(0, height):
        for x in range(0, width):
            if img[y][x] < 20:
                line = [] 
                for theta in range(0, 180):
                    p = int(x * math.cos(math.radians(theta)) + y * math.sin(math.radians(theta)))
                    accumulator[theta][p] += 1
                    if (accumulator[theta][p] > line_length) and (p not in lines[:, 0]) and (theta not in lines[:, 1]):
                        lines = np.vstack((lines, np.array([p, theta])))

    lines = np.delete(lines, [0, 1], axis=0)
    return lines

def line_intersection(p, theta, img):
    h, w = img.shape[:2]
    out = []
    theta = math.radians(theta)
    intersect = [int(round(p / math.sin(theta))), int(round((p - w * math.cos(theta)) / math.sin(theta))), int(round(p / math.cos(theta))),
                 int(round((p - h * math.sin(theta)) / math.cos(theta)))]
    if (intersect[0] > 0) and (intersect[0] < h):
        out.append((0, intersect[0]))
    if (intersect[1] > 0) and (intersect[1] < h):
        out.append((w, intersect[1]))

    if (intersect[2] > 0) and (intersect[2] < w):
        out.append((intersect[2], 0))
    if (intersect[3] > 0) and (intersect[3] < w):
        out.append((intersect[3], h))
    return out

def thresh(image):
	h, w = image.shape
	Hist = np.zeros([256], np.uint64)
	for i in range(h):
	    for j in range(w):
	        Hist[image[i][j]] += 1

	maxLoc = np.where(Hist == np.max(Hist))
	first = maxLoc[0][0]
	meas = np.zeros([256], np.float32)
	
	for k in range(256):
	    meas[k] = pow(k-first, 2) * Hist[k]
	maxLoc2 = np.where(meas == np.max(meas))
	second = maxLoc2[0][0]
	thresh = 0
	if first > second:
	    temp = Hist[int(second):int(first)]
	    minLoc = np.where(temp == np.min(temp))
	    thresh = second + minLoc[0][0] + 1
	else:
	    temp = Hist[int(first):int(second)]
	    minLoc = np.where(temp == np.min(temp))
	    thresh = first + minLoc[0][0] + 1

	threshImage = image.copy()

	threshImage[threshImage > thresh] = 255
	threshImage[threshImage <= thresh] = 0
	return threshImage



#img = getimg()
img = cv2.imread("3.jpg")
#Rimg, Gimg, Bimg = cv2.split(img)


#homework1--------------------------------------------------------------------------------------------
#question1--------------------------------------------------------------------------------------------
'''sobelimg = sobeldetect(img)
cv2.imshow('sobel', sobelimg)
cv2.waitKey(0)'''

#question2--------------------------------------------------------------------------------------------
'''perwittimg = perwittdetect(img)
cv2.imshow('perwitt', perwittimg)
cv2.waitKey(0)'''

#question3--------------------------------------------------------------------------------------------
'''robertsimg = robertsdetect(img)
cv2.imshow('roberts', robertsimg)
cv2.waitKey(0)'''

#question4--------------------------------------------------------------------------------------------
'''laplaceimg = laplacedetect(img)
cv2.imshow('laplace', laplaceimg)
cv2.waitKey(0)'''

#homework2--------------------------------------------------------------------------------------------
#question1--------------------------------------------------------------------------------------------
'''img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lines = Hough_lines(img)
print("lines:",lines)

for i in lines:
        points = line_intersection(i[0], i[1], img)
        print(points)
        cv2.line(img, points[0], points[1], [100])

cv2.imshow('image', img)
cv2.waitKey(0)'''

#homework2--------------------------------------------------------------------------------------------
#question1--------------------------------------------------------------------------------------------
'''img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = thresh(img)
cv2.imshow('image', img)
cv2.waitKey(0)'''
