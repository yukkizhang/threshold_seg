import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import skimage.io as io


img = cv2.imread(r"D:\2PM_2023\segmentation\therehold_method\dataset\PPFE_elastin\2d_image/6_coll_0035-1.tif", 0)

#傅里叶变换
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
# 将频谱低频从左上角移动到中心位置
fshift = np.fft.fftshift(dft)

# res1 = 20 * np.log(cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))
res1 = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
plt.imshow(res1,plt.cm.gray)
plt.show()
# io.imsave("fft.png", res1)

# img_fft = cv2.imread(r"D:\2PM_2023\segmentation\therehold_method\fft.png")
img_fft = res1
img_fft[93:96, 171:345] = 0
img_fft[160:163, 186:324] = 0
img_fft[349:352, 188:325] = 0
img_fft[417:419, 193:329] = 0
plt.imshow(img_fft, plt.cm.gray)
plt.show()
print("end")

#傅里叶逆变换
ishift = np.fft.ifftshift(dft)
# iimg = cv2.idft(img_fft)
iimg = cv2.idft(ishift)
res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])

plt.imshow(res,plt.cm.gray)
plt.show()
print("end")