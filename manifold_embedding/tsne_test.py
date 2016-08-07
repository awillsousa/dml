import numpy as np
import matplotlib.pyplot as plt
import gzip
import cPickle as pickle
from sklearn import manifold

filename = "../data/mnist.pkl.gz"
f = gzip.open(filename, 'rb')
train_data = pickle.load(f)[0]
f.close()

imgs_start_index = 0
imgs_range = 5

tsne = manifold.TSNE()
train_data_tnse = tsne.fit_transform(np.array(train_data[0][:1000]))

for i in range(imgs_start_index, imgs_start_index + imgs_range):
        plt.subplot(5, 2, 2*i+1)
        plt.title(str('index = '+str(i)) + '- target =' + str(train_data[1][i]))
        fig = plt.imshow(train_data[0][i].reshape((28, 28)), cmap=plt.cm.binary)
        fig.axes.get_xaxis().set_ticks([])
        fig.axes.get_yaxis().set_ticks([])
        plt.subplot(5, 2, 2*i + 2)
        plt.title(str('index = ' + str(i)) + '- target =' + str(train_data[1][i]))
        fig = plt.imshow(train_data[0][i].reshape((28, 28)), cmap=plt.cm.binary)
        fig.axes.get_xaxis().set_ticks([])
        fig.axes.get_yaxis().set_ticks([])
plt.tight_layout()
plt.show()