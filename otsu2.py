import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# img = cv.imread(r"D:\2PM_2023\segmentation\therehold_method\test.png", cv.IMREAD_GRAYSCALE)
img = cv.imread(r"D:\2PM_2023\segmentation\therehold_method\dataset\PPFE_elastin\2d_image/init_0000.tif", cv.IMREAD_GRAYSCALE)


assert img is not None, "file could not be read, check with os.path.exists()"
# # global thresholding--阈值设置为127，255表示二值图中一个阈值为0，另一个阈值为255
# ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# # Otsu's thresholding
# ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# # Otsu's thresholding after Gaussian filtering
# blur = cv.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# # plot all the images and their histograms
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
#           'Original Noisy Image','Histogram',"Otsu's Thresholding",
#           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# for i in range(3):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()

import cv2


image = img
maxval = 255
otsuThe = 0
otsuThe, dst_Otsu = cv2.threshold(image, otsuThe, maxval, cv2.THRESH_OTSU)
cv2.imshow('Otsu', dst_Otsu)

# cv2.imwrite('otsu2_result.png', dst_Otsu)
cv2.imwrite('otsu2_init_result.png', dst_Otsu)

cv2.waitKey(0)
cv2.destroyAllWindows()
