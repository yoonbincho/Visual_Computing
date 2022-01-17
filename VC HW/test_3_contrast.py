import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plot_grid_size = (4, 3)
def plotImage(index1, index2, index3, img, img_rgb, title):
    plt.subplot(plot_grid_size[0],plot_grid_size[1], index1)
    plt.imshow(img, cmap='gray')
    plt.axis('off'), plt.title(title) 
    plt.subplot(plot_grid_size[0],plot_grid_size[1], index3)
    plt.imshow(img_rgb, cmap='gray')
    plt.axis('off'), plt.title(title)   
    plt.subplot(plot_grid_size[0],plot_grid_size[1], index2)
    hist,bins = np.histogram(img.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256]), plt.title(title) 
    plt.legend(('cdf','histogram'), loc = 'upper left')

img = cv.imread("picture.jpg")
img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
img_yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

img_he = cv.equalizeHist(img)
img_heYuv = img_yuv.copy()
img_heYuv[:, :, 0] = cv.equalizeHist(img_heYuv[:,:,0]) 
img_heRgb = cv.cvtColor(img_heYuv, cv.COLOR_YUV2RGB)

ahe = cv.createCLAHE(clipLimit=255, tileGridSize=(8,8))
img_ahe = ahe.apply(img)
img_ahaYuv = img_yuv.copy()
img_ahaYuv[:, :, 0] = ahe.apply(img_ahaYuv[:,:,0])
img_aheRgb = cv.cvtColor(img_ahaYuv, cv.COLOR_YUV2RGB)

clahe = cv.createCLAHE(clipLimit = 2, tileGridSize=(8,8))
img_clahe = clahe.apply(img)
img_claheYuv = img_yuv.copy()
img_claheYuv[:, :, 0] = clahe.apply(img_claheYuv[:,:,0])
img_claheRgb = cv.cvtColor(img_claheYuv, cv.COLOR_YUV2RGB)

plotImage(1, 2, 3, img, img_rgb,"original")
plotImage(4, 5, 6, img_he, img_heRgb, "HE") 
plotImage(7, 8, 9, img_ahe, img_aheRgb, "AHE")
plotImage(10, 11, 12, img_clahe, img_claheRgb, "CLAHE")

plt.show()