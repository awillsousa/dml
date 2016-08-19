import cv2
import numpy as np
import timeit
import logging
import os
import cPickle as pickle
from scipy import ndimage
import fli
import scipy.ndimage.interpolation as scipint

from os import listdir
from os import remove
from os.path import isfile, join
path = '../data/custom/'

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def get_blurred_vector(vector, blur=1):
    return ndimage.gaussian_filter(vector.reshape(28, 28), blur).flatten()

def get_blurred_sets(set_x, set_y, blur=1):
    return np.apply_along_axis(get_blurred_vector, axis=1, arr=set_x, blur=blur), set_y

def get_rotated_vector(vector, angle=0):
    return scipint.rotate(vector.reshape(28, 28), angle, order=0, reshape=False).flatten()

def get_rotated_sets(set_x, set_y, angle=0):
    return np.apply_along_axis(get_rotated_vector, axis=1, arr=set_x, angle=angle), set_y

#deletes the blurred image files in the path directori
def remove_blur_files(path=path):
    files = [f for f in listdir(path) if isfile(join(path, f)) and 'blur' in f]
    for file in files:
        remove(path+file)

#creates blurred versions of the files in the path directory
def create_blur_files(path=path, blur=1):
    files = [f for f in listdir(path) if isfile(join(path, f)) and 'blur' not in f]
    for file in files:
        test_img = fli.processImg(path, file, flatten=False)
        i = file.find('.')
        img_blurred = np.invert(ndimage.gaussian_filter(test_img.reshape((28, 28)), blur))
        cv2.imwrite(path + file[:i] + '_blur_a.png', img_blurred)

def save_model(params, epoch=-1 , best_validation_loss=-1, test_score=-1, namestub='test'
               ,randomInit=False, add_blurs=False, testrun=False, logfilename='testLog.log',
               endrun=False, annotation = ''):
    blur = ''
    last = ''
    if randomInit:
        init_1 = '_rand'
    else:
        init_1 = '_zero'
    if add_blurs:
        blur = '_blur'
    if testrun:
        last = '_test'

    savedFileName = namestub + str(epoch) + init_1 + annotation + blur + last +'.pkl'
    gg = open(savedFileName, 'wb')
    pickle.dump(params, gg, protocol=pickle.HIGHEST_PROTOCOL)
    gg.close()
    print(('Best model params saved as ' + savedFileName
           + ' with test score %f %%') % (test_score * 100.))

    time = timeit.default_timer()
    if not os.path.isfile(logfilename):
        open(logfilename, 'w+')
    if not os.path.getsize(logfilename) > 0:
        logging.info('end_time;epoch;filenamme;best_validation_score;test_score')
    logging.info(str(time) + ';' + str(epoch) + ';' + savedFileName +
                 ';' + str(best_validation_loss * 100) +
                 ';' + str(test_score * 100.))
    if endrun:
        logging.info('-----------------------------------------')

def load_params(filename):
    gg = open(filename, 'rb')
    params = pickle.load(gg)
    gg.close()
    return params

def epoch_from_filename(filename):
    return filter(str.isdigit, filename)
