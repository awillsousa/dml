import gzip
import cPickle as pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc
import sys
sys.path.insert(0, '../mlp_test')
import fli

#test_img = fli.processImg('../data/custom/', 'test_0.png', flatten=False)
#test_img = cv2.imread('../data/custom/test_0.png')
#img_blurred = ndimage.gaussian_filter(test_img, 1)
#cv2.imwrite('../data/transform/' + 'trans_test_0_blur.png', img_blurred)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#plot_image = np.concatenate((test_img , img_blurred), axis=1)

#plt.imshow(plot_image)

#plt.show()

from os import listdir
from os import remove
from os.path import isfile, join
path = '../data/custom/'
files = [f for f in listdir(path) if isfile(join(path, f)) and 'blur' in f]
for file in files:
    remove(path+file)


blur=1

files = [f for f in listdir(path) if isfile(join(path, f)) and 'blur' not in f]
for file in files:
    test_img_value = filter(str.isdigit, file)
    test_img = fli.processImg(path, file, flatten=False)
    i = file.find('.')
    img_blurred = np.invert(ndimage.gaussian_filter(test_img.reshape((28, 28)), blur))
    cv2.imwrite('../data/custom/' + file[:i] + '_blur_a.png', img_blurred)

# blur = 1
# path = '../data/custom/'
# file = 'own_0.png'
# test_img_value = filter(str.isdigit, file)
# test_img = flymod.processImg(path, file)
# i = file.find('.')
# img_blurred = ndimage.gaussian_filter(test_img.reshape((28, 28)), blur)
#
# #plot_image = np.concatenate((test_img , img_blurred), axis=1)
#
# plot_image = test_img
#
# plt.imshow(plot_image)
#
# filepath = "../data/savedImage_0.p"
#
# test_saved = pickle.load( open(filepath, "rb" ) )
#
# test_x_saved, test_y_saved = test_saved
#
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# # plt.imshow(train_x[0].reshape((28, 28)), cmap = cm.Greys_r)
# # plt.show()
#
# #example_index=np.random.randint(10000)
# #example_index=9904
# #test_x, test_y = test_set
# plt.title(test_y_saved,  fontsize=24)
# plt.imshow(test_x_saved.reshape((28, 28)), cmap = cm.Greys_r)
# plt.show()
#
# blur = 1
# path = '../data/custom/'
# file = 'own_0.png'
# test_img_value = filter(str.isdigit, file)
# test_img = flymod.processImg(path, file)
# i = file.find('.')
# img_blurred = ndimage.gaussian_filter(test_img.reshape((28, 28)), blur)
#misc.imsave(path +'test_blur.png', img_blurred)


# path = '../data/custom/'
# file = 'own_0.png'
# test_img = fli.processImg(path, file)
# img_blurred = ndimage.gaussian_filter(test_img.reshape((28, 28)), blur)
# misc.imsave(path + 'test_blur.png', img_blurred)

