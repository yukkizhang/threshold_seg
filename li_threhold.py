import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage.filters.thresholding import _cross_entropy
import cv2

# cell = data.cell()
# camera = data.camera()
# camera = cv2.imread(r"D:\2PM_2023\segmentation\therehold_method\test.png", cv2.IMREAD_GRAYSCALE)
camera = cv2.imread(r"D:\2PM_2023\segmentation\therehold_method\dataset\PPFE_elastin\2d_image/4_0006.tif", cv2.IMREAD_GRAYSCALE)


thresholds = np.arange(np.min(camera) + 1.5, np.max(camera) - 1.5)
entropies = [_cross_entropy(camera, t) for t in thresholds]

optimal_camera_threshold = thresholds[np.argmin(entropies)]

fig, ax = plt.subplots(1, 3, figsize=(8, 3))

ax[0].imshow(camera, cmap='gray')
ax[0].set_title('image')
ax[0].set_axis_off()

ax[1].imshow(camera > optimal_camera_threshold, cmap='gray')
ax[1].set_title('thresholded')
ax[1].set_axis_off()

ax[2].plot(thresholds, entropies)
ax[2].set_xlabel('thresholds')
ax[2].set_ylabel('cross-entropy')
ax[2].vlines(optimal_camera_threshold,
             ymin=np.min(entropies) - 0.05 * np.ptp(entropies),
             ymax=np.max(entropies) - 0.05 * np.ptp(entropies))
ax[2].set_title('optimal threshold')

fig.tight_layout()

print('The brute force optimal threshold is:', optimal_camera_threshold)
print('The computed optimal threshold is:', filters.threshold_li(camera))

plt.show()