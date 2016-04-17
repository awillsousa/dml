import network3_1
from network3_1 import Network
from network3_1 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3_1.load_data_shared()
mini_batch_size = 10
import six.moves.cPickle
import theano.tensor as T
import numpy as np
import theano
# net = Network([
# 	FullyConnectedLayer(n_in=784, n_out=100),
#         SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size, test_data)
# net.SGD(training_data, 10, mini_batch_size, 0.1,validation_data, test_data)

#-------------------




gg = open('savedSelflayers.saved', 'rb')
layers = six.moves.cPickle.load(gg)
gg.close()

net = Network([
	layers[-2],
        layers[-1] ], mini_batch_size)
net.outputAccuracy(test_data)