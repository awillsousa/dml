# Cf. http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

import numpy as np
import matplotlib.pyplot as plt
import gzip
import cPickle as pickle
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from scipy.spatial import ConvexHull
from time import time
import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist


showAll = True
plotVertexImages = False
testlen = 1000
start=30000

target_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
colors = target_values / 10.


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    plt.figure(figsize=(8, 7))
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    points = [[] for k in range(len(target_values))]
    for i in range(X.shape[0]):
        for j in range(len(target_values)):
            if (y[i] == target_values[j]):
                plt.text(X[i, 0], X[i, 1], str(y[i]),
                         color=plt.cm.Set1(colors[j]),
                         fontdict={'weight': 'bold', 'size': 9})
                points[j].append([X[i, 0], X[i, 1]])
                break
    if plotVertexImages:
        vertexindexes =[]
        for j in range(len(target_values)):
            the_points = np.asarray(points[j])
            hull = ConvexHull(np.asarray(the_points))
            for simplex in hull.simplices:
                plt.plot(the_points[simplex, 0], the_points[simplex, 1], color=plt.cm.Set1(colors[j]), ls='-')
            vertexindexes.append(hull.vertices)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    if plotVertexImages:
        plot_vertex_images(title = "Vertex Images - "+ title, rangeimg=vertexindexes)

#Plot images of the vertex digits
def plot_vertex_images(title = None, rangeimg=None):
    nr_vertex_images = 0
    for k in range(len(rangeimg)):
        nr_vertex_images += len(rangeimg[k])
    print("nr of vertex images = " + str(nr_vertex_images))
    breakk = False
    n_img_per_row = 7
    targ_val = 0
    the_index = -1
    img = np.zeros((28 * n_img_per_row, 28 * n_img_per_row))
    for i in range(n_img_per_row):
        ix = 28 * i
        for j in range(n_img_per_row):
            the_index += 1
            if (the_index == len(rangeimg[targ_val])) :
                targ_val += 1
                the_index = 0
            if (i * n_img_per_row + j == nr_vertex_images):
                breakk = True
                break
            iy = 28 * j
            img[ix:ix + 28, iy:iy + 28] = X_data[rangeimg[targ_val][the_index]].reshape((28, 28))
        if breakk:
            break
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

#----------------------------------------------------------------------
# Projection on to the first 2 linear discriminant components
def ldp(X, y ,nr_components=2):
    print("Computing Linear Discriminant Analysis projection")
    X2 = X.copy()
    X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
    return discriminant_analysis.LinearDiscriminantAnalysis(n_components=nr_components).fit_transform(X2, y)

#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
def rp(X, nr_components=2):
    rp = random_projection.SparseRandomProjection(n_components=nr_components, random_state=42)
    return rp.fit_transform(X)



#----------------------------------------------------------------------
# Projection on to the first 2 principal components
def pca(X, nr_components=2):
    print("Computing PCA projection")
    return decomposition.TruncatedSVD(n_components=nr_components).fit_transform(X)


#----------------------------------------------------------------------
# Modified Locally linear embedding of the digits dataset
def lle(X, nr_components=2, n_neighbors = 30):
    print("Computing modified LLE embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=nr_components,
                                          method='modified')
    X_mlle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    return X_mlle


#----------------------------------------------------------------------
# MDS  embedding of the digits dataset
def md5(X, nr_components=2):
    print("Computing MDS embedding")
    clf = manifold.MDS(n_components=nr_components, n_init=1, max_iter=100)
    t0 = time()
    X_mds = clf.fit_transform(X)
    print("Done. Stress: %f" % clf.stress_)
    return X_mds

#----------------------------------------------------------------------
# Random Trees embedding of the digits dataset
def trt(X, nr_components=2):
    print("Computing Totally Random Trees embedding")
    hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                           max_depth=5)
    t0 = time()
    X_transformed = hasher.fit_transform(X)
    pca = decomposition.TruncatedSVD(n_components=nr_components)
    return pca.fit_transform(X_transformed)

#----------------------------------------------------------------------
# Spectral embedding of the digits dataset
def cse(X, nr_components=2):
    print("Computing Spectral embedding")
    embedder = manifold.SpectralEmbedding(n_components=nr_components, random_state=0,
                                          eigen_solver="arpack")
    return embedder.fit_transform(X)


def tsne(X, nr_components=2):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=nr_components, init='random', random_state=0)
    return tsne.fit_transform(X)


if __name__ == '__main__':
    train_data = load_mnist()[0]

    chosens = [index for index in range(start, start + testlen) if train_data[1][index] in target_values]

    indexes = np.asarray([i for i in chosens])
    X_data = np.asarray([train_data[0][i] for i in chosens])
    y_data = np.asarray([train_data[1][i] for i in chosens])


    if showAll:
        t0 = time()
        plot_embedding(tsne(X_data), y_data,
                       "t-SNE embedding of the digits (time %.2fs)" %
                       (time() - t0))
        t0 = time()
        plot_embedding(cse(X_data), y_data,
                       "Spectral embedding of the digits (time %.2fs)" %
                       (time() - t0))
        t0 = time()
        plot_embedding(trt(X_data), y_data,
                       "Random forest embedding of the digits (time %.2fs)" %
                       (time() - t0))
        t0 = time()
        plot_embedding(md5(X_data), y_data,
                       "MDS embedding of the digits (time %.2fs)" %
                       (time() - t0))

        plot_embedding(lle(X_data), y_data,
                       "Modified Locally Linear Embedding of the digits (time %.2fs)" %
                       (time() - t0))
        t0 = time()
        plot_embedding(pca(X_data), y_data,
                       "Principal Components projection of the digits (time %.2fs)" %
                       (time() - t0))
        plot_embedding(rp(X_data), y_data, "Random Projection of the digits")
    t0 = time()
    plot_embedding(ldp(X_data, y_data), y_data,
                   "Linear Discriminant projection of the digits (time %.2fs)" %
                   (time() - t0))
    plt.show()