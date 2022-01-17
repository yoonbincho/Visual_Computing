import cv2 as cv
import numpy as np
from scipy import ndimage
from numpy.core.fromnumeric import shape
from matplotlib import pyplot as plt
from matplotlib.pyplot import title

rgb_img = cv.cvtColor(cv.imread('peach.jpg'), cv.COLOR_BGR2RGB)
img = cv.imread('peach.jpg',cv.IMREAD_GRAYSCALE)

def gray_img(index, img, title):
    plt.subplot(2,4, index), plt.imshow(img, cmap='gray'), plt.axis('off')
    plt.title(title)

def blur(im): # blur 
    a = np.array([[2, 4, 5, 4, 2],[4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]])* (1/159)
    i = cv.filter2D(im, -1 ,a)
    return i
    
def sobel(i): # Sobel Filter
    sobelx_64 = cv.Sobel(i,cv.CV_32F,1,0,ksize=3)
    absolute_x_64 = np.absolute(sobelx_64)
    sobelx_8u1 = absolute_x_64/absolute_x_64.max()*255
    sobelx_8u = np.uint8(sobelx_8u1)

    sobely_64 = cv.Sobel(i,cv.CV_32F,0,1,ksize=3)
    absolute_y_64 = np.absolute(sobely_64)
    sobely_8u1 = absolute_y_64/absolute_y_64.max()*255
    sobely_8u = np.uint8(sobely_8u1)
    
    gradient = np.hypot(sobelx_8u, sobely_8u)
    gradient = gradient / gradient.max()*255
    gradient = np.uint8(gradient)
    
    theta = np.arctan2(sobely_64, sobelx_64)
    return (sobelx_8u + sobely_8u, gradient , theta)

def edge_thinning(i, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
  
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
               #0º
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #45º
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #90º
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #135º
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass   
    return Z

def threshold (i, select, lowThresholdRatio= 0.03, highThresholdRatio = 0.15):
    highThreshold = i.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = i.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    zero = 255
    weak = 150
    high = 190
    
    high_a, high_b = np.where(i >= highThreshold)
    zeros_a, zeros_b = np.where(i < lowThreshold)
    
    weak_a, weak_b = np.where((i <= highThreshold) & (i >= lowThreshold))
    
    if select == 1:
        res[high_a, high_b] = high
        res[weak_a, weak_b] = 0
        res[zeros_a, zeros_b] = 0
    elif select == 2:
        res[weak_a, weak_b] = weak
    else:
        res[high_a, high_b] = high
        res[weak_a, weak_b] = weak
        res[zeros_a, zeros_b] = zero
    
    return (res, high, weak)

def cmap(index, img, title, cmtype):
    plt.subplot(2,4, index), plt.imshow(img, cmtype), plt.axis('off')
    plt.title(title)
    
def hystresis(img, weak, high):
    M, N = img.shape  
    for a in range(1, M-1):
        for b in range(1, N-1):
            if (img[a,b] == weak):
                try:
                    if ((img[a+1, b-1] == high) or (img[a+1,b] == high) or (img[a+1, b+1] == high)
                        or (img[a, b-1] == high) or (img[a, b+1] == high)
                        or (img[a-1, b-1] == high) or (img[a-1,b] == high) or (img[a-1, b+1] == high)):
                        img[a, b] = 25
                    else:
                        img[a, b] = weak
                except IndexError as e:
                    pass
    return img

def final_img(img, weak, high):
    M, N = img.shape  
    for a in range(1, M-1):
        for b in range(1, N-1):
            if (img[a,b] == weak):
                try:
                    if ((img[a+1, b-1] == high) or (img[a+1,b] == high) or (img[a+1, b+1] == high)
                        or (img[a, b-1] == high) or (img[a, b+1] == high)
                        or (img[a-1, b-1] == high) or (img[a-1,b] == high) or (img[a-1, b+1] == high)):
                        img[a, b] = 25
                    else:
                        img[a, b] = 255
                except IndexError as e:
                    pass
    return img

gray_img(1, rgb_img, 'input image')
gray_img(2, img, 'gray scale image')    
img = blur(img)
gray_img(3, img, 'stage1 result')
img = sobel(img)[0]
gray_img(5, img, 'stage2 result')
img = edge_thinning(img, sobel(img)[2])
gray_img(6, img, 'stage3 result')

img = threshold(img, 3)[0]
weak = threshold(img, 3)[2]
high = threshold(img,3)[1]
cmap(7, hystresis(img,weak,high), 'stage4 result', 'gist_ncar')
cmap(8, final_img(img,weak,high), 'final output image', 'gist_ncar')

plt.show()