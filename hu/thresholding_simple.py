from __future__ import division

import numpy as np
import gzip
import cPickle as pickle
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist

thresholds = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]

train_data, verificatio_data, test_data = load_mnist()

index_1 = 7
img_arr_1 = train_data[0][index_1].reshape((28, 28))
threshos =[]
img = np.uint8(img_arr_1 * 255)
threshos.append(img)
i = 1;
for tr in thresholds:
    ret, thresh = cv2.threshold(img, 127, 255, tr)
    threshos.append(thresh)

thresh = ['img', 'thresh1', 'thresh2', 'thresh3', 'thresh4', 'thresh5']

for i in xrange(6):
    plt.subplot(2, 3, i + 1), plt.imshow(threshos[i], 'gray')
    plt.title(thresh[i])

plt.show()