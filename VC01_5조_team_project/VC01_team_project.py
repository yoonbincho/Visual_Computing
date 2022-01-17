import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def plot_img(rows, cols, index, img, title):
    ax = plt.subplot(rows,cols,index)
    if(len(img.shape) == 3):
        ax_img = plt.imshow(img[...,::-1]) # same as img[:,:,::-1]), RGB image is displayed without cv.cvtColor
    else:
        ax_img = plt.imshow(img, cmap='gray')
    plt.axis('on')
    if(title != None): plt.title(title) 
    return ax_img, ax
    
def display_untilKey(Pimgs, Titles, file_out = False):
    for img, title in zip(Pimgs, Titles):
        cv.imshow(title, img)
        if file_out == True:
            cv.imwrite(title + ".jpg", img)
    cv.waitKey(0)

# Initiate SIFT detector
# find the keypoints and descriptors with SIFT
# search detectAndCompute
def match_keypoints(img1, img2) : 
    sift = cv.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    flann = cv.FlannBasedMatcher({"algorithm":1, "trees":5}, {"checks":50})
    matches = flann.knnMatch(des1, des2, k=2)
    
    return kp1, des1, kp2, des2, matches    

# Store all the good matches as per Lowe's ratio test.
def update_good_correpondences(ratio_dist, match):
    good_correspondences = [] 
    for m,n in match:
        if m.distance/n.distance < ratio_dist:
            good_correspondences.append(m)
    return good_correspondences

    
# For settings 
img1 = cv.imread("01.jpg")
img2 = cv.imread("02.jpg")
img3 = cv.imread("03.jpg")
img4 = cv.imread("04.jpg")
img5 = cv.imread("05.jpg")

img1 = cv.resize(img1, (400,300))
img2 = cv.resize(img2, (400,300))
img3 = cv.resize(img3, (400,300))
img4 = cv.resize(img4, (400,300))
img5 = cv.resize(img5, (400,300))

imgs = [img1, img2, img3, img4, img5]
kps = []
des = [] 
good_correspondences = []
matches = []
img_matches = []

for i in range(4) :
    kp1, des1, kp2, des2, match = match_keypoints(imgs[i], imgs[i+1])
    good_correspondence = update_good_correpondences(0.7, match)
    img_matches_now = cv.drawMatches(imgs[i],kp1,imgs[i+1],kp2,good_correspondence,None,matchColor=(0,255,0),singlePointColor=None,matchesMask=None,flags=2)
    
    good_correspondences.append(good_correspondence)
    matches.append(match)
    kps.append(kp1)
    if (i==3) : kps.append(kp2)
    des.append(des1)
    if (i==3) : des.append(des2)
    img_matches.append(img_matches_now)



# Slider and button
from matplotlib.widgets import Slider, Button, RadioButtons

fig = plt.figure(1)
ax = []
ax_imgs = []
tx = []

fig.canvas.mpl_connect('close_event', lambda e : plt.close('all'))
for i in range (4) :
    title = "mathing result (img" + (str)(i+1) + " & img" + (str)(i+2) + ")"
    ax_img, ax_now = plot_img(2,2 ,i+1,img_matches[i],title)
    ax.append(ax_now)
    ax_imgs.append(ax_img)
    tx_now = ax_now.text(0.05, 0.95, "# good correspondences: " + str(len(good_correspondences[i])), transform=ax_now.transAxes, fontsize=14,
        verticalalignment='top', bbox={'boxstyle':'round', 'facecolor':'wheat', 'alpha':0.5})
    tx.append(tx_now)
slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_ratio = Slider(slider_ax, 'ratio dist', 0.1, 1.0, valinit=0.7, valstep=0.1)

# ratio dist에 따라 각 list에 저장된 (kp, des, correspondences pair..) 요소들 update
def update(val):
    for i in range (4) :
        good_correspondences_now = update_good_correpondences(slider_ratio.val, matches[i])
        img_matches_now = cv.drawMatches(imgs[i],kps[i],imgs[i+1],kps[i+1],good_correspondences_now,None,matchColor=(0,255,0),singlePointColor=None,matchesMask=None,flags=2)
        update_draw(img_matches_now, ax_imgs[i], tx[i], good_correspondences_now)
        good_correspondences[i] = good_correspondences_now
        img_matches[i] = img_matches_now
    
# ratio dist에 따라 바뀌는 good correspondences pair를 보여주기 위함 
def update_draw(matches, ax, tx, good_correspondences) : 
    matches = cv.cvtColor(matches, cv.COLOR_BGR2RGB)
    ax.set_data(matches)
    tx.set_text("# good correspondences: " + str(len(good_correspondences)))
    fig.canvas.draw_idle() 
    
slider_ratio.on_changed(update)

button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(button_ax, 'Stitch', color='lightgoldenrodyellow', hovercolor='0.975')



# For Stitching
def stitch(event):
    results = []   # 이전에 stitching 된 결과물을 계속해서 저장 
    plt.figure(2)
    
    # stitching for img1 & img2 
    # 원본(source) 좌표 배열, 결과(destination) 좌표 배열
    kp1, kp2 = kps[0], kps[1] 
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_correspondences[0] ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_correspondences[0] ]).reshape(-1,1,2)
    H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

    # 영상물의 잘림을 방지하기 위해 넉넉하게 row, col값을 잡아줌 
    stitch_plane_rows = imgs[0].shape[0] + mask.shape[0]
    stitch_plane_cols = imgs[0].shape[1] + imgs[1].shape[1]
    result1 = cv.warpPerspective(imgs[1], H, (stitch_plane_cols, stitch_plane_rows), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)
        
    result2 = np.zeros((stitch_plane_rows, stitch_plane_cols,3), np.uint8)
    result2[0:imgs[0].shape[0], 0:imgs[0].shape[1]] = imgs[0]
        
    and_img = cv.bitwise_and(result1, result2)
    and_img_gray = cv.cvtColor(and_img, cv.COLOR_BGR2GRAY)
    th, mask1 = cv.threshold(and_img_gray, 1, 255, cv.THRESH_BINARY)
        
    plot_img(3, 4, 1, result1, None)
    plot_img(3, 4, 5, result2, None)
    plot_img(3, 4, 9, mask1, None)
        
    result = np.zeros((stitch_plane_rows, stitch_plane_cols,3), np.uint8)
    for y in range(stitch_plane_rows):
        for x in range(stitch_plane_cols):
            mask_v = mask1[y, x]
            if(mask_v > 0):
                result[y, x] = np.uint8(result1[y,x] * 0.5 + result2[y,x] * 0.5)
            elif(np.any(result2[y,x])):
                result[y, x] = result2[y,x]
            else:
                result[y, x] = result1[y,x]
    results.append(result)
    
    # Stitching for img3, img4, img5
    for i in range (1, 4) :
        kp1, des1, kp2, des2, match = match_keypoints(results[i-1], imgs[i+1])
        good_correspondences_now = update_good_correpondences(slider_ratio.val, match)
        
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_correspondences_now ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_correspondences_now ]).reshape(-1,1,2)
        H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
        
        stitch_plane_rows += mask.shape[0]
        stitch_plane_cols += imgs[i+1].shape[1]
        result1 = cv.warpPerspective(imgs[i+1], H, (stitch_plane_cols, stitch_plane_rows), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)
        
        result2 = np.zeros((stitch_plane_rows, stitch_plane_cols,3), np.uint8)
        result2[0:result.shape[0], 0:result.shape[1]] = result
        
        and_img = cv.bitwise_and(result1, result2)
        and_img_gray = cv.cvtColor(and_img, cv.COLOR_BGR2GRAY)
        th, mask1 = cv.threshold(and_img_gray, 1, 255, cv.THRESH_BINARY)
        
        plot_img(3, 4, i+1, result1, None)
        plot_img(3, 4, i+5, result2, None)
        plot_img(3, 4, i+9, mask1, None)
        
        result = np.zeros((stitch_plane_rows, stitch_plane_cols,3), np.uint8)
        for y in range(stitch_plane_rows):
            for x in range(stitch_plane_cols):
                mask_v = mask1[y, x]
                if(mask_v > 0):
                    result[y, x] = np.uint8(result1[y,x] * 0.5 + result2[y,x] * 0.5)
                elif(np.any(result2[y,x])):
                    result[y, x] = result2[y,x]
                else:
                    result[y, x] = result1[y,x]
        results.append(result)
    
    plt.figure(3)
    result_final = result[0:450, 0:1280].copy()
    plot_img(1,1,1,result_final,None)
    plt.show()

button.on_clicked(stitch)
plt.show()