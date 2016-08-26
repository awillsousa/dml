from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from time import time
import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist

from manifold_embedding_1 import ldp, tsne, lle, cse, trt, pca, md5, trt

showAll=True


# Scale and visualize the embedding vectors
def plot_embedding(X, y,  title=None):
    colors = target_values / 10.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    for i in range(X.shape[0]):
        for j in range(len(target_values)):
            if (y[i] == target_values[j]):
                ax.text(X[i, 0], X[i, 1], X[i, 2], str(y[i]),
                         color=plt.cm.Set1(colors[j]),
                         fontdict={'weight': 'bold', 'size': 9})
                break
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_zlabel('LD3')
    if title is not None:
        plt.title(title)


testlen = 1000
start=30000
target_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


if __name__ == '__main__':
    train_data = load_mnist()[0]


    chosens = [index for index in range(start, start + testlen) if train_data[1][index] in target_values]

    indexes = np.asarray([i for i in chosens])
    X_data = np.asarray([train_data[0][i] for i in chosens])
    y_data = np.asarray([train_data[1][i] for i in chosens])


    if showAll:
        t0 = time()
        plot_embedding(tsne(X_data, nr_components=3), y_data,
                       "t-SNE embedding of the digits (time %.2fs)" %
                       (time() - t0))
        t0 = time()
        plot_embedding(cse(X_data, nr_components=3), y_data,
                       "Spectral embedding of the digits (time %.2fs)" %
                       (time() - t0))
        t0 = time()
        plot_embedding(trt(X_data, nr_components=3), y_data,
                       "Random forest embedding of the digits (time %.2fs)" %
                       (time() - t0))
        t0 = time()
        plot_embedding(md5(X_data, nr_components=3), y_data,
                       "MDS embedding of the digits (time %.2fs)" %
                       (time() - t0))

        plot_embedding(lle(X_data, nr_components=3), y_data,
                       "Modified Locally Linear Embedding of the digits (time %.2fs)" %
                       (time() - t0))
        t0 = time()
        plot_embedding(pca(X_data, nr_components=3), y_data,
                       "Principal Components projection of the digits (time %.2fs)" %
                       (time() - t0))
        plot_embedding(trt(X_data, nr_components=3), y_data, "Random Projection of the digits")
    t0 = time()
    plot_embedding(ldp(X_data, y_data, nr_components = 3), y_data,
                   "Linear Discriminant projection of the digits 3D(time %.2fs)" %
                   (time() - t0))
    plt.show()