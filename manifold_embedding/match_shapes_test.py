from __future__ import division

import numpy as np
import gzip
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from cv2 import matchShapes


filename = "../data/mnist.pkl.gz"
f = gzip.open(filename, 'rb')
train_data, verificatio_data, test_data = pickle.load(f)
f.close()

index_1 = 7
index_2 = 10
img_arr_1 = train_data[0][index_1].reshape((28, 28))
img_val_1 = train_data[1][index_1]
img_arr_2 = train_data[0][index_2].reshape((28, 28))
img_val_2 = train_data[1][index_2]
import scipy.misc
scipy.misc.imsave('outfile.png', img_arr_1)
ret, thresh = cv2.threshold(np.uint8(img_arr_1*255),127,255,cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(np.uint8(img_arr_2 * 255), 127, 255,0)
contours,hierarchy = cv2.findContours(thresh, 2, 1)
cnt1 = contours[0]
contours,hierarchy = cv2.findContours(thresh2, 2, 1)
cnt2 = contours[0]
match_I1 = matchShapes(cnt1, cnt2, cv2.cv.CV_CONTOURS_MATCH_I1,0)
plt.suptitle("Comparing two images - match_I1 = "+ str(match_I1))
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