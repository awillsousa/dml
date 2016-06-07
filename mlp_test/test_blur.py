import gzip
import cPickle as pickle
import numpy as np
from scipy import ndimage
import fli

with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

chosen_index = 1250

img = test_set[0][chosen_index].reshape((28, 28))
img_blurred = ndimage.gaussian_filter(img, 1)
test_y_chosen = test_set[1][chosen_index]

plot_image = np.concatenate((img, img_blurred), axis=1)


import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.title(test_y_chosen,  fontsize=24)
plt.imshow(plot_image, cmap = cm.Greys_r)
plt.show()

from os import listdir
from os.path import isfile, join
blur = 1
path = '../data/custom/'
files = [f for f in listdir(path) if isfile(join(path, f)) and 'blur' not in f]
for file in files:
    test_img_value = filter(str.isdigit, file)
    test_img = fli.processImg(path, file)
    i = file.find('.')
    img_blurred = ndimage.gaussian_filter(test_img.reshape((28, 28)), blur)
    plt.imshow(img_blurred, cmap=cm.Greys_r)
    plt.show()
    # gg = open(path + file[:i] + '_blur_' + str(blur) +'.png', "wb")
    # pickle.dump(img_blurred, gg, protocol=pickle.HIGHEST_PROTOCOL)
    # gg.close()

