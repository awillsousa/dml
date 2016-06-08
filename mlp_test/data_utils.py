import cv2
import numpy as np
from scipy import ndimage, misc
import fli

from os import listdir
from os import remove
from os.path import isfile, join
path = '../data/custom/'

#deletes the blurred image files in the path directori
def remove_blur_files(path=path):
    files = [f for f in listdir(path) if isfile(join(path, f)) and 'blur' in f]
    for file in files:
        remove(path+file)
blur=1

#creates blurred versions of the files in the path directory
def create_blur_files(path=path):
    files = [f for f in listdir(path) if isfile(join(path, f)) and 'blur' not in f]
    for file in files:
        test_img = fli.processImg(path, file, flatten=False)
        i = file.find('.')
        img_blurred = np.invert(ndimage.gaussian_filter(test_img.reshape((28, 28)), blur))
        cv2.imwrite(path + file[:i] + '_blur_a.png', img_blurred)
