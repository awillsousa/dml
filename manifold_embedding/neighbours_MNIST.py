# TSNE-plotting for different distances
from __future__ import division

import numpy as np

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist

imgpath='../data/pics/'

train_len = 2000
start=0

# 'mahalanobis'

target_values = np.array([1])
metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
           'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
            'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
           'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
           'yule']

# The metrics 'mahalanobis', 'seuclidean', 'cosine' are not directly usable

train_data, validation_data, test_data = load_mnist()

chosens = [(index, train_data[1][index]) for index in range(start, start + train_len) if train_data[1][index] in target_values]

sorted_chosens = np.asarray(sorted(chosens, key=lambda target: target[1]))
X_data = train_data[0][start:start + train_len]
y_data = train_data[1][start:start + train_len]

X_test = test_data[0]
y_test = test_data[1]

len_test = len(test_data[1])
from scipy.spatial.distance import cdist

def closest_node(node, nodes, distance = 'euclidean'):
    return cdist([node], nodes, metric = distance).argmin()

results = []
for met in metrics:
    nerrors = 0
    for i in range(len(X_test)):
        closest_train_pic_index = closest_node(X_test[i], X_data, distance = met)
        if y_data[closest_train_pic_index] != y_test[i]:
            # print("Value " + str(y_test[i]) + " at index " + str(i) + \
            #       " different from value " + str(y_data[closest_train_pic_index]) + \
            #       " of closest neighbour at index " + str(closest_train_pic_index))
            nerrors += 1
    print(str(nerrors) + " mismatches out of " + str(train_len))
    errorperc = nerrors * 100 / len_test
    print (met + ' - ' +"{:4.2f}".format(errorperc) + '% error rate ')
    results.append([met,errorperc])

results = sorted(results, key= lambda val: val[1])
ind = np.arange(len(results))
width = 0.35       # the width of the bars: can also be len(x) sequence
plt.xticks(ind + width/2., zip(*results)[0], rotation=40)
p1 = plt.bar(ind, zip(*results)[1], width, color = ['magenta', 'green', 'yellow', 'red', 'maroon'])
plt.title('Prediction scores on MNIST test by distance on ' + str(train_len) + ' training sample')
for rect, val in zip(p1.patches, zip(*results)[1]):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2, height , "{:4.2f}".format(val), ha='center', va='bottom')
plt.show()




