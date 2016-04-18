import network3_2
from network3_2 import Network
from network3_2 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3_2.load_data_shared()
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




gg = open('savedSelf.saved', 'rb')
savedNet= six.moves.cPickle.load(gg)
gg.close()



savedNet.outputAccuracy(test_data)