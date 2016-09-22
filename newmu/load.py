# see https://github.com/Newmu/Theano-Tutorials
import numpy as np
import os
import cPickle as pickle
import gzip

datasets_dir = './'

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def load_mnist(mnist_path="../data/mnist.pkl.gz"):
    data_dir, data_file = os.path.split(mnist_path)
    if (not os.path.isfile(mnist_path)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, mnist_path)
    f = gzip.open(mnist_path, 'rb')
    dataset = pickle.load(f)
    f.close()
    return dataset

def mnist(ntrain=50000,ntest=10000,onehot=True):
	dataset  = load_mnist()
	trX = dataset[0][0]
	trY = dataset[0][1]
	teX = dataset[2][0]
	teY = dataset[2][1]

	trX = trX/255.
	teX = teX/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	return trX,teX,trY,teY

def nomnist(range = -1):
	datasetpath = '../data/notMNIST.pickle'

	gg = open(datasetpath, 'rb')
	the_data = pickle.load(gg)
	gg.close()

	train_dataset = the_data['train_dataset'][:range]
	valid_dataset = the_data['valid_dataset']
	train_labels = the_data['train_labels'][:range]
	test_dataset = the_data['test_dataset']
	valid_labels = the_data['valid_labels']
	test_labels = the_data['test_labels']

	return train_dataset.reshape((-1, 28 * 28 )), one_hot(train_labels, 10), valid_dataset.reshape((-1, 28 * 28 )),\
		   one_hot(valid_labels , 10), test_dataset.reshape((-1, 28 * 28 )), one_hot(test_labels , 10)

