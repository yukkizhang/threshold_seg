import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage.filters.thresholding import _cross_entropy
import cv2
import skimage.io as io


# cell = cv2.imread(r"D:\2PM_2023\segmentation\therehold_method\dataset\PPFE_elastin\2d_image/6_coll_0036.tif", cv2.IMREAD_GRAYSCALE)
# cell = cv2.imread(r"D:\2PM_2023\segmentation\therehold_method\dataset\PPFE_elastin\2d_image/Inverse FFT of 6_coll_0035-1.tif", cv2.IMREAD_GRAYSCALE)
cell = cv2.imread(r"D:\2PM_2023\segmentation\therehold_method\dataset\PPFE_elastin\2d_image/6_coll_0035-1.tif", cv2.IMREAD_GRAYSCALE)

cell = cv2.GaussianBlur(cell,(5,5),0)

thresholds = np.arange(np.min(cell) + 1.5, np.max(cell) - 1.5)


iter_thresholds2 = []

opt_threshold2 = filters.threshold_li(cell, initial_guess=64,
                                      iter_callback=iter_thresholds2.append)

thresholds2 = np.arange(np.min(cell) + 1.5, np.max(cell) - 1.5)
entropies2 = [_cross_entropy(cell, t) for t in thresholds]
iter_entropies2 = [_cross_entropy(cell, t) for t in iter_thresholds2]

fig, ax = plt.subplots(1, 3, figsize=(8, 3))

ax[0].imshow(cell, cmap='gray')
ax[0].set_title('image')
ax[0].set_axis_off()

result = cell > opt_threshold2
ax[1].imshow(result, cmap='gray')
# ax[1].imshow(cell > opt_threshold2, cmap='gray')
ax[1].set_title('thresholded')
ax[1].set_axis_off()

ax[2].plot(thresholds2, entropies2, label='all threshold entropies')
ax[2].plot(iter_thresholds2, iter_entropies2, label='optimization path')
ax[2].scatter(iter_thresholds2, iter_entropies2, c='C1')
ax[2].legend(loc='upper right')

# io.imsave('li2_result.png', result)
# cv2.imwrite('li2_result.png', result)

io.imsave('coll_li2_result.png', result)

plt.show()