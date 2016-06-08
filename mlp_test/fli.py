# from https://github.com/openmachinesblog/tensorflow-mnist/blob/master/mnist.py by

import cv2
import numpy as np
import math
from scipy import ndimage


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)
    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx,shifty

def processImg(img_dirpath, img_filename , save="True", save_path="../data/transform/", flatten = True):
    formattedImg = cv2.imread(img_dirpath + "/" + img_filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    formattedImg = cv2.resize(255 - formattedImg, (28, 28))
    (thresh, formattedImg) = cv2.threshold(formattedImg, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    while np.sum(formattedImg[0]) == 0:
        formattedImg = formattedImg[1:]

    while np.sum(formattedImg[:, 0]) == 0:
        formattedImg = np.delete(formattedImg, 0, 1)

    while np.sum(formattedImg[-1]) == 0:
        formattedImg = formattedImg[:-1]

    while np.sum(formattedImg[:, -1]) == 0:
        formattedImg = np.delete(formattedImg, -1, 1)

    rows, cols = formattedImg.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        # first cols than rows
        formattedImg = cv2.resize(formattedImg, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        # first cols than rows
        formattedImg = cv2.resize(formattedImg, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    formattedImg = np.lib.pad(formattedImg, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = getBestShift(formattedImg)
    shifted = shift(formattedImg, shiftx, shifty)
    formattedImg = shifted

    # save the processed images
    if save:
        cv2.imwrite(save_path + 'trans_' + img_filename, formattedImg)
    if flatten:
        return formattedImg.flatten() / 255.0
    return formattedImg