import numpy as np
import gzip
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage.interpolation as scipint
import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import get_rotated_sets, load_mnist

# def get_rotated_vector(vector, angle=0):
#     return scipint.rotate(vector.reshape(28, 28), angle, order=0, reshape=False).flatten()
#
# def get_rotated_sets(set_x, set_y, angle=0):
#     return np.apply_along_axis(get_rotated_vector, axis=1, arr=set_x, angle=angle), set_y

index_1 = 4
rotangle = -30

data_set = load_mnist()[0]
data_set_rotated = get_rotated_sets(data_set[0], data_set[1], rotangle)


img_arr_1 = data_set_rotated[0][index_1].reshape((28, 28))
img_val_1 = data_set_rotated[1][index_1]

rotArr = scipint.rotate(img_arr_1, rotangle, order=0, reshape = False)

plt.title(str(img_val_1))
fig = plt.imshow(img_arr_1, cmap=cm.binary)
fig.axes.get_xaxis().set_ticks([])
fig.axes.get_yaxis().set_ticks([])

plt.show()