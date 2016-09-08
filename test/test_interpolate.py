import pywt
import sys
import numpy as np
from scipy.ndimage.interpolation import affine_transform
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist
from skimage import transform as tf



test_data = load_mnist()[2]

chosen_index = 7

test_x_chosen = test_data[0][chosen_index]
test_y_chosen = test_data[1][chosen_index]

transm = np.eye(28, k=0) + np.eye(28, k=1)

pic_arr = test_x_chosen.reshape((28, 28))

pic_trans = np.dot(pic_arr, transm)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.subplot(2 , 1, 1)
plt.imshow(pic_arr, cmap = cm.Greys_r,interpolation='nearest')
plt.subplot(2 , 1, 2)
plt.imshow(pic_trans, cmap = cm.Greys_r,interpolation='nearest')
plt.show()