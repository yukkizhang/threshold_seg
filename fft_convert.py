import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def put(path):
    img = cv2.imread(path, 1)
    # img = cv2.imread(os.path.join(base, path), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape[:2]
    # 傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 将频谱低频从左上角移动至中心位置
    dft_shift = np.fft.fftshift(dft)
    # 频谱图像双通道复数转换为0-255区间
    res1 = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(dft_shift)
    iimg = cv2.idft(ishift)
    res2 = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    # 图像顺时针旋转60度
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -60, 1)
    rot = cv2.warpAffine(img, M, (rows, cols))
    # 傅里叶变换
    dft3 = cv2.dft(np.float32(rot), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift3 = np.fft.fftshift(dft3)
    res3 = 20 * np.log(cv2.magnitude(dft_shift3[:, :, 0], dft_shift3[:, :, 1]))
    # 傅里叶逆变换
    ishift3 = np.fft.ifftshift(dft_shift3)
    iimg3 = cv2.idft(ishift3)
    res4 = cv2.magnitude(iimg3[:, :, 0], iimg3[:, :, 1])

    # 图像向右平移
    H = np.float32([[1, 0, 200], [0, 1, 0]])
    tra = cv2.warpAffine(img, H, (rows, cols))
    # 傅里叶变换
    dft2 = cv2.dft(np.float32(tra), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift2 = np.fft.fftshift(dft2)
    res5 = 20 * np.log(cv2.magnitude(dft_shift2[:, :, 0], dft_shift2[:, :, 1]))
    # 傅里叶逆变换
    ishift2 = np.fft.ifftshift(dft_shift2)
    iimg2 = cv2.idft(ishift2)
    res6 = cv2.magnitude(iimg2[:, :, 0], iimg2[:, :, 1])
    # 输出结果
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(331), plt.imshow(img, plt.cm.gray), plt.title('原图灰度图像'),plt.axis('off')
    plt.subplot(332),plt.imshow(res1,plt.cm.gray),plt.title('傅里叶变换'),plt.axis('off')
    plt.subplot(333),plt.imshow(res2,plt.cm.gray),plt.title('傅里叶反变换'),plt.axis('off')
    plt.subplot(334),plt.imshow(rot,plt.cm.gray),plt.title('图像旋转'),plt.axis('off')
    plt.subplot(335),plt.imshow(res3,plt.cm.gray),plt.title('傅里叶变换'),plt.axis('off')
    plt.subplot(336),plt.imshow(res4,plt.cm.gray),plt.title('傅里叶反变换'),plt.axis('off')
    plt.subplot(337),plt.imshow(tra,plt.cm.gray),plt.title('图像平移'),plt.axis('off')
    plt.subplot(338),plt.imshow(res5,plt.cm.gray),plt.title('傅里叶变换'),plt.axis('off')
    plt.subplot(339),plt.imshow(res6,plt.cm.gray),plt.title('傅里叶反变换'),plt.axis('off')

    # plt.savefig('1.new.jpg')
    plt.show()

# 处理函数，要传入路径
put(r'../image/image3.jpg')
