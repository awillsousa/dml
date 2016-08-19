from __future__ import division

import numpy as np
import gzip
import cPickle as pickle
import cv2


thresholds = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]

filename = "../data/mnist.pkl.gz"
f = gzip.open(filename, 'rb')
train_data, verificatio_data, test_data = pickle.load(f)
f.close()

index_1 = 7
img_arr_1 = train_data[0][index_1].reshape((28, 28))

import numpy as np
import cv2

thresholds = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]


img = np.uint8(img_arr_1 * 255)
the_image = cv2.resize(img.copy(), (200, 200))
for tr in thresholds:
    ret,thresh = cv2.threshold(img.copy(),127,255,tr)
    contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours,-1,(255,0,0),3)
    resized_image = cv2.resize(img, (200, 200))
    the_image = np.concatenate((the_image, resized_image), axis=1)
cv2.imshow("Contours",the_image)
cv2.waitKey()
