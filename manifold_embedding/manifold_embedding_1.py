# Cf. http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

import numpy as np
import matplotlib.pyplot as plt
import gzip
import cPickle as pickle
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from scipy.spatial import ConvexHull
from time import time

filename = "../data/mnist.pkl.gz"
f = gzip.open(filename, 'rb')
test_data = pickle.load(f)[0]
f.close()

showAll = False
plotVertexImages = True
testlen = 5000
start=0

target_values = np.array([7, 1, 2])
colors = target_values / 10.
chosens = [index for index in range(start, start+testlen) if test_data[1][index] in target_values]

indexes = np.asarray([i for i in chosens])
X = np.asarray([test_data[0][i] for i in chosens])
y = np.asarray([test_data[1][i] for i in chosens])

n_samples, n_features = X.shape
n_neighbors = 30

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    plt.figure()
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
            img[ix:ix + 28, iy:iy + 28] = X[rangeimg[targ_val][the_index]].reshape((28, 28))
        if breakk:
            break
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)


#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
if showAll:
    rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
    X_projected = rp.fit_transform(X)
    plot_embedding(X_projected, "Random Projection of the digits")


#----------------------------------------------------------------------
# Projection on to the first 2 principal components
if showAll:
    print("Computing PCA projection")
    t0 = time()
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    plot_embedding(X_pca,
                   "Principal Components projection of the digits (time %.2fs)" %
                   (time() - t0))

#----------------------------------------------------------------------
# Projection on to the first 2 linear discriminant components
print("Computing Linear Discriminant Analysis projection")
X2 = X.copy()
X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
t0 = time()
X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
plot_embedding(X_lda,
               "Linear Discriminant projection of the digits (time %.2fs)" %
               (time() - t0))

#----------------------------------------------------------------------
# Modified Locally linear embedding of the digits dataset
if showAll:
    print("Computing modified LLE embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                          method='modified')
    t0 = time()
    X_mlle = clf.fit_transform(X)
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding(X_mlle,
                   "Modified Locally Linear Embedding of the digits (time %.2fs)" %
                   (time() - t0))




#----------------------------------------------------------------------
# MDS  embedding of the digits dataset
if showAll:
    print("Computing MDS embedding")
    clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
    t0 = time()
    X_mds = clf.fit_transform(X)
    print("Done. Stress: %f" % clf.stress_)
    plot_embedding(X_mds,
                   "MDS embedding of the digits (time %.2fs)" %
                   (time() - t0))

#----------------------------------------------------------------------
# Random Trees embedding of the digits dataset
if showAll:
    print("Computing Totally Random Trees embedding")
    hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                           max_depth=5)
    t0 = time()
    X_transformed = hasher.fit_transform(X)
    pca = decomposition.TruncatedSVD(n_components=2)
    X_reduced = pca.fit_transform(X_transformed)

    plot_embedding(X_reduced,
                   "Random forest embedding of the digits (time %.2fs)" %
                   (time() - t0))

#----------------------------------------------------------------------
# Spectral embedding of the digits dataset
if showAll:
    print("Computing Spectral embedding")
    embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                          eigen_solver="arpack")
    t0 = time()
    X_se = embedder.fit_transform(X)

    plot_embedding(X_se,
                   "Spectral embedding of the digits (time %.2fs)" %
                   (time() - t0))

if showAll:
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X_tsne,
                   "t-SNE embedding of the digits (time %.2fs)" %
                   (time() - t0))

plt.show()