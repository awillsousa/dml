from __future__ import division

import numpy as np
import gzip
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

thresholds = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]

filename = "../data/mnist.pkl.gz"
f = gzip.open(filename, 'rb')
train_data, verificatio_data, test_data = pickle.load(f)
f.close()

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