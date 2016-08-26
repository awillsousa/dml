from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from cv2 import matchShapes
from skimage.measure import compare_ssim
import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist


train_data, verificatio_data, test_data = load_mnist()


index_1 = 7
index_2 = 10
img_arr_1 = train_data[0][index_1].reshape((28, 28))
img_val_1 = train_data[1][index_1]
img_arr_2 = train_data[0][index_2].reshape((28, 28))
img_val_2 = train_data[1][index_2]
import scipy.misc
ret, thresh_1 = cv2.threshold(np.uint8(img_arr_1 * 255).copy(), 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(np.uint8(img_arr_2 * 255).copy(), 127, 255,0)
contours,hierarchy = cv2.findContours(thresh_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt1 = contours[0]
contours,hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt2 = contours[0]
formatt = "{:4.2f}"
match_I1 = formatt.format(matchShapes(cnt1, cnt2, cv2.cv.CV_CONTOURS_MATCH_I1,0))
match_I2 = formatt.format(matchShapes(cnt1, cnt2, cv2.cv.CV_CONTOURS_MATCH_I2,0))
match_I3 = formatt.format(matchShapes(cnt1, cnt2, cv2.cv.CV_CONTOURS_MATCH_I3,0))
plt.suptitle("Comparing two images - match_I1 = "+ match_I1+ " - match_I2 = "+ match_I2
             + " - match_I3 = " + match_I3 +'\n Structural similarity = ' + formatt.format(compare_ssim(img_arr_1, img_arr_2)))
plt.subplot(1, 2, 1)
plt.title(str(img_val_1))
fig = plt.imshow(img_arr_1, cmap=cm.binary)
fig.axes.get_xaxis().set_ticks([])
fig.axes.get_yaxis().set_ticks([])
plt.subplot(1, 2, 2)
plt.title(str(img_val_2))
fig = plt.imshow(img_arr_2 , cmap=cm.binary)
fig.axes.get_xaxis().set_ticks([])
fig.axes.get_yaxis().set_ticks([])
plt.tight_layout()
plt.show()