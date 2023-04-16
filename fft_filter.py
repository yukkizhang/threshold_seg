import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import skimage.io as io

# #读取图像
# img = cv.imread(r"D:\2PM_2023\segmentation\therehold_method\dataset\PPFE_elastin\2d_image/6_coll_0035-1.tif", 0)

# #傅里叶变换
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)

# #设置高通滤波器
# rows, cols = img.shape
# crow,ccol = int(rows/2), int(cols/2)
# fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

# #傅里叶逆变换
# ishift = np.fft.ifftshift(fshift)
# iimg = np.fft.ifft2(ishift)
# iimg = np.abs(iimg)

# #设置字体
# matplotlib.rcParams['font.sans-serif']=['SimHei']

# #显示原始图像和高通滤波处理图像
# plt.subplot(121), plt.imshow(img, 'gray'), plt.title(u'(a)原始图像')
# plt.axis('off')
# plt.subplot(122), plt.imshow(iimg, 'gray'), plt.title(u'(b)结果图像')
# plt.axis('off')
# plt.show()

# io.imsave('coll_fftg_result.png', iimg)



import cv2
# import numpy as np
# from matplotlib import pyplot as plt

#读取图像
# img = cv2.imread(r"D:\2PM_2023\segmentation\therehold_method\dataset\PPFE_elastin\2d_image/6_coll_0035-1.tif", 0)

img = cv2.imread(r"D:\2PM_2023\segmentation\therehold_method\dataset\PPFE_elastin\2d_image/6_coll_0036.tif", 0)

#傅里叶变换
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
# 将频谱低频从左上角移动到中心位置
fshift = np.fft.fftshift(dft)

# res1 = 20 * np.log(cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))
# plt.imshow(res1,plt.cm.gray)
# plt.show()
# io.imsave("fft.png", res1)


#设置低通滤波器
rows, cols = img.shape
crow,ccol = int(rows/2), int(cols/2) #中心位置
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

#掩膜图像和频谱图像乘积
f = fshift * mask
print(f.shape, fshift.shape, mask.shape)


#傅里叶逆变换
ishift = np.fft.ifftshift(f)
iimg = cv2.idft(ishift)
res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])

#显示原始图像和低通滤波处理图像
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(122), plt.imshow(res, 'gray'), plt.title('Result Image')
plt.axis('off')
plt.show()

io.imsave('coll_fftd_result.png', iimg)
