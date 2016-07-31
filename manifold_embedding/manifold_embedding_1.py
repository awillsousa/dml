# Cf. http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.htmlfrom time import time

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
testlen = 5000
start=0

vals = np.array([9, 5, 1])
colors = vals / 10.
chosens = [index for index in range(start, start+testlen) if test_data[1][index] in vals]

X = np.asarray([test_data[0][i] for i in chosens])
y = np.asarray([test_data[1][i] for i in chosens])
# X = digits.data
#
n_samples, n_features = X.shape
n_neighbors = 30

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    points=[[],[],[]]
    for i in range(X.shape[0]):
        for j in range(len(vals)):
            if (y[i] == vals[j]):
                plt.text(X[i, 0], X[i, 1], str(y[i]),
                         color=plt.cm.Set1(colors[j]),
                         fontdict={'weight': 'bold', 'size': 9})
                points[j].append([X[i, 0], X[i, 1]])
                break

    hulls = []
    for j in range(len(vals)):
        points[j] = np.asarray(points[j])
        hull = ConvexHull(np.asarray(points[j]))
        for simplex in hull.simplices:
            plt.plot(points[j][simplex, 0], points[j][simplex, 1], color=plt.cm.Set1(colors[j]), ls='-')

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

#----------------------------------------------------------------------
# Plot images of the digits
# n_img_per_row = 20
# img = np.zeros((28 * n_img_per_row, 28 * n_img_per_row))
# for i in range(n_img_per_row):
#     ix = 10 * i + 1
#     for j in range(n_img_per_row):
#         iy = 10 * j + 1
#         img[ix:ix + 28, iy:iy + 28] = X[i * n_img_per_row + j].reshape((28, 28))
#
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.title('A selection from the 64-dimensional digits dataset')


#----------------------------------------------------------------------
# Random 2D projection using a random unitary matrix
if showAll == True:
    print("Computing random projection")
    rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
    X_projected = rp.fit_transform(X)
    plot_embedding(X_projected, "Random Projection of the digits")


#----------------------------------------------------------------------
# Projection on to the first 2 principal components
if showAll == True:
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
if showAll == True:
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
if showAll == True:
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
if showAll == True:
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
if showAll == True:
    print("Computing Spectral embedding")
    embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                          eigen_solver="arpack")
    t0 = time()
    X_se = embedder.fit_transform(X)

    plot_embedding(X_se,
                   "Spectral embedding of the digits (time %.2fs)" %
                   (time() - t0))

if showAll == True:
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X_tsne,
                   "t-SNE embedding of the digits (time %.2fs)" %
                   (time() - t0))

plt.show()