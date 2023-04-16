import cv2 as cv
import numpy as np
np.seterr(invalid='ignore') #之所以加这一句是因为下面除法的时候会除零报错
import matplotlib.pyplot as plt


def otsu3layers(imagepath):  # 显示每一个灰度级的像素数概率
    img = cv.imread(imagepath, 0)  # 以灰度图形式读取图片

    high, wide = img.shape
    pixels = cv.calcHist([img], [0], None, [256], [0, 256])  # 计算每个灰度级中所含像素数，返回的是一个（256,1）的数组
    p = pixels / (high * wide)  # 获得每个灰度级中像素数占总像素数比例则我们获得了p_i的一个向量
    x=np.linspace(1,256,256) #灰度级我们定义为从0到255好像不合适这样做平均灰度级时会忽略第一个数据，所以我们定义为从1到256
    x=x.reshape(256,1)  # 这一步是因为我们需要的不是秩为1的向量，可以不加这一句输出看看shape的区别，但这样我们得到了（256,1）的灰度级向量

    maxvar=0
    th1=0
    th2=0
    cnt=0
    for k1 in range(1,257):
        for k2 in range(k1+1,257):
            w0 = np.sum(p[0:k1])
            w1 = np.sum(p[k1:k2])
            w2 = np.sum(p[k2:256])  # 注意python是左闭右开np.sum(p[0:256])=1
            u0 = np.dot(x[0:k1].T, p[0:k1])
            u1 = np.dot(x[k1:k2].T, p[k1:k2])
            u2 = np.dot(x[k2:256].T, p[k2:256])
            W = np.array([[w0, w1, w2]]).T  # 注意是两层中括号，用np.array组成的向量可以转置,输出矩阵大小等等操作
            U = np.array([[int(u0), int(u1), int(u2)]]).T
            UT = np.sum(U)
            temp = (U - UT * W) * (U - UT * W) / W
            varbetween = np.sum(temp)
            if varbetween>maxvar:
                cnt=cnt+1
                maxvar=varbetween
                th1=k1
                th2=k2
                print("第"+str(cnt)+"次找到最佳阈值："+str(th1)+","+str(th2)+"此时类间方差为"+str(maxvar))
    # mean1 = int(np.dot(x[0:th1].T, p[0:th1]))
    # mean2 = int(np.dot(x[th1:th2].T, p[th1:th2]))
    # mean3 = int(np.dot(x[th2:256].T, p[th2:256]))
    for row in range(0,high):
        for col in range(0,wide):
            if img[row, col]<th1:
                img[row, col]=0
            elif img[row, col]>=th1 and img[row, col]<=th2:
                img[row, col] =127
            else:
                img[row, col] = 255
    cv.imshow("2th3gray",img)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # otsu3layers(r"D:\2PM_2023\segmentation\therehold_method\dataset\PPFE_elastin\2d_image/4_0006.tif")
    otsu3layers(r"D:\2PM_2023\segmentation\therehold_method\test.png")
