import cv2
import numpy as np

def gaussian_pyramid(img, n):
    low = img.copy()
    gaussian_pyrmid = [low]
    i = 0
    while i < n:
        low = cv2.pyrDown(low)
        gaussian_pyrmid.append(np.float32(low))
        i += 1
    return gaussian_pyrmid
 
def laplacian_pyramid(gaussian_pyramid):
    laplacian_t = gaussian_pyramid[-1]
    n = len(gaussian_pyramid) - 1
    laplacian_pyr = [laplacian_t]
    i = n
    while i > 0:
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyramid[i-1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
        i -= 1
    return laplacian_pyr
 
def blended(x,y,mask_pyramid):
    LS = []
    for la,lb,mask in zip(x,y,mask_pyramid):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS
 
def blended_finish(laplacian_pyramid):
    laplacian_t = laplacian_pyramid[0]
    laplacian_l = [laplacian_t]
    num = len(laplacian_pyramid) - 1
    i = 0
    while i < num:
        size = (laplacian_pyramid[i + 1].shape[1], laplacian_pyramid[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_t, dstsize=size)
        laplacian_t = cv2.add(laplacian_pyramid[i+1], laplacian_expanded)
        laplacian_l.append(laplacian_t)
        i += 1
    return laplacian_l

if __name__ == '__main__':
    hand_img = cv2.imread('hand.jpg')
    hand_img = cv2.resize(hand_img, (1800, 1000))
    eye_img = cv2.imread('eye.jpg')
    eye_img = cv2.resize(eye_img, (1800, 1000))
    mask = np.zeros((1000,1800,3), dtype='float32')
    mask[400:600,600:1200,:] = (1,1,1)
    num_levels = 7
    gaussian_hand = gaussian_pyramid(hand_img, num_levels)
    laplacian_hand = laplacian_pyramid(gaussian_hand)
    gaussian_eye = gaussian_pyramid(eye_img, num_levels)
    laplacian_eye = laplacian_pyramid(gaussian_eye)
    mask_f = gaussian_pyramid(mask, num_levels)
    mask_f.reverse()
    bledning = blended(laplacian_hand,laplacian_eye,mask_f)
    final  = blended_finish(bledning)
    cv2.imwrite('masks.jpg',final[num_levels])
    images = cv2.imread("masks.jpg", cv2.IMREAD_ANYCOLOR)
    resize_img = cv2.resize(images, (300,300))
    cv2.imshow("Mask", resize_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    