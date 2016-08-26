# TSNE-plotting for different distances

import gzip
import cPickle as pickle

import seaborn as sns

import numpy as np

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist


imgpath='../data/pics/'

testlen = 2000
start=0

target_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# The metrics 'mahalanobis', 'seuclidean', 'cosine' are not directly usable

metrics = ['euclidean', 'l1', 'l2', 'manhattan','braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']


data_set = load_mnist()[0]


chosens = [(index,data_set[1][index])  for index in range(start, start + testlen) if data_set[1][index] in target_values]

sorted_chosens = np.asarray(sorted(chosens, key=lambda target: target[1]))
X_data = np.asarray(data_set[0][sorted_chosens[:,0]])
y_data = np.asarray([data_set[1][sorted_chosens[:,0]]])[0]

# Random state.
RS = 20150101

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


for metric_type in metrics:
    digits_proj = TSNE(random_state=RS, metric=metric_type).fit_transform(X_data)
    scatter(digits_proj, y_data)
    plt.savefig(imgpath + 'MNIST-tsne-'+metric_type+'.png', dpi=120)
    print('MNIST-tsne-'+metric_type+'.png saved')