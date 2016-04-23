import gzip
import theano
import cPickle
import numpy as np
import network3_2

test_saved = cPickle.load( open("savedImage.p", "rb" ) )
test_x_saved, test_y_saved = test_saved

filename = "../data/mnist.pkl.gz"
f = gzip.open(filename, 'rb')
test_data = cPickle.load(f)[2]
f.close()

test_data[0][0] = test_x_saved
test_data[1][0] = test_y_saved #The result is 1

#test_data[1][0] = 4 #The result is 0

shared_x = theano.shared(np.asarray(test_data[0], dtype=theano.config.floatX), borrow=True)
shared_y = theano.shared(np.asarray(test_data[1], dtype=theano.config.floatX), borrow=True)
test_data_shared = shared_x, theano.tensor.cast(shared_y, "int32")

gg = open('savedSelf.p', 'rb')
savedNet = cPickle.load(gg)
gg.close()
out_1 = savedNet.singleOutputPrediction(test_data_shared, 0)
print "The result is %s" % out_1

