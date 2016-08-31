import pywt
import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist

mode = 'sym2'
level = None

direction = ['h', 'v', 'd']

test_data = load_mnist()[2]

chosen_index = 7

test_x_chosen = test_data[0][chosen_index]
test_y_chosen = test_data[1][chosen_index]

pic_arr = test_x_chosen.reshape((28, 28))

pic_wts = pywt.wavedec2(pic_arr, mode, level=level)

length = len(pic_wts)

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


plt.subplot(length , 3, 2)
ax = plt.subplot(length , 3, 2)
ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
plt.title(' mode = ' + mode ,  fontsize=18)
plt.imshow(pic_arr, cmap = cm.Greys_r,interpolation='nearest')
for i in range(1, length):
    for j in range(1, 4):
        ax = plt.subplot(length,3, 3 * i + j)
        ax.set_title(direction[j-1] + ' - ' + str(i))
        ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
        ax.imshow(pic_wts[i][j-1], cmap = cm.Greys_r, interpolation='nearest')
plt.tight_layout( h_pad=0.1)
plt.show()