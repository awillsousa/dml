import numpy as np
import gzip
import cPickle as pickle
import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage.interpolation as scipint
import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import get_rotated_sets

# def get_rotated_vector(vector, angle=0):
#     return scipint.rotate(vector.reshape(28, 28), angle, order=0, reshape=False).flatten()
#
# def get_rotated_sets(set_x, set_y, angle=0):
#     return np.apply_along_axis(get_rotated_vector, axis=1, arr=set_x, angle=angle), set_y

index_1 = 4
rotangle = -30

filename = "../data/mnist.pkl.gz"
f = gzip.open(filename, 'rb')
data_set = pickle.load(f)[0]
data_set_rotated = get_rotated_sets(data_set[0], data_set[1], rotangle)
f.close()

#rot_data = scipint.rotate(data_set[0], rotangle, order=0, reshape = False)

img_arr_1 = data_set_rotated[0][index_1].reshape((28, 28))
img_val_1 = data_set_rotated[1][index_1]

rotArr = scipint.rotate(img_arr_1, rotangle, order=0, reshape = False)

plt.title(str(img_val_1))
fig = plt.imshow(img_arr_1, cmap=cm.binary)
fig.axes.get_xaxis().set_ticks([])
fig.axes.get_yaxis().set_ticks([])

plt.show()