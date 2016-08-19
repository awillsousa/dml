from __future__ import division

import numpy as np
import gzip
import cPickle as pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

train_len = 1000
start=0

the_colors = [ 'blue', 'green', 'red', 'cyan', 'magenta',  'yellow', 'darkblue', 'lawngreen', 'orange', 'violet']

showSequence = True

filename = "../data/mnist.pkl.gz"
f = gzip.open(filename, 'rb')
data_set = pickle.load(f)[0]
f.close()

target_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

chosens = [(index,data_set[1][index])  for index in range(start, start + train_len) if data_set[1][index] in target_values]

sorted_chosens = np.asarray(sorted(chosens, key=lambda target: target[1]))
X_data = np.asarray(data_set[0][sorted_chosens[:,0]])
y_data = np.asarray([data_set[1][sorted_chosens[:,0]]])[0]

clrs = [the_colors[k] for i in range(train_len) for k in target_values if y_data[i] == k]
if not showSequence:
    params = {'legend.fontsize': 6}
    plt.rcParams.update(params)
patches = []
for k in target_values:
    patch = mpatches.Patch(color=the_colors[k], label=str(target_values[k]))
    patches.append(patch)


for l in range(7):
    plt.title('Hu moment ' + str(l))
    if not showSequence:
        plt.subplot(4, 2, l+1)
    mom_1 = []
    for index in range(train_len):
        img_arr = X_data[index]
        img_val = y_data[index]
        mom_1.append(cv2.HuMoments(cv2.moments(img_arr))[l])
    plt.legend(handles=patches)
    plt.scatter(range(train_len),mom_1, color=clrs)
    if showSequence:
        plt.show()
if not showSequence:
    plt.show()


