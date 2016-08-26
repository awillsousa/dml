import numpy as np
import gzip
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage.interpolation as scipint
import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist

data_set = load_mnist()[0]


index_1 = 4
rotangle = 30

img_arr_1 = data_set[0][index_1].reshape((28, 28))
img_val_1 = data_set[1][index_1]

rotArr = scipint.rotate(img_arr_1, rotangle, order=0, reshape = False)


plt.subplot(1, 2, 1)
plt.title(str(img_val_1))
fig = plt.imshow(img_arr_1, cmap=cm.binary)
fig.axes.get_xaxis().set_ticks([])
fig.axes.get_yaxis().set_ticks([])


plt.subplot(1, 2, 2)
plt.title("Rotated scipy")
fig = plt.imshow(rotArr, cmap=cm.binary)
fig.axes.get_xaxis().set_ticks([])
fig.axes.get_yaxis().set_ticks([])

plt.show()

