import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
import cPickle
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = Network([
	FullyConnectedLayer(n_in=784, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 10, mini_batch_size, 0.1,validation_data, test_data)
gg = open('net3.p', 'wb')
cPickle.dump(net, gg, protocol=cPickle.HIGHEST_PROTOCOL)
gg.close()