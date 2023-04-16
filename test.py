import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage.filters.thresholding import _cross_entropy
import cv2
import skimage.io as io

img = io.imread(r"D:\2PM_2023\segmentation\therehold_method\dataset\PPFE_elastin/6-1-1.tif")

new_img = img[:,list(range(0,img.shape[1],2))]

io.imsave("test_result2.tif",new_img)
