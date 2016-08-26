import formatAndLoadImage as fli
import gzip
import theano
import cPickle
import numpy as np
import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist


test_img = fli.processImg('../data/custom/','own_3.png')

test_data = load_mnist()[2]

test_data[0][0] = test_img
test_data[1][0] = 4 #The result is 1

#test_data[1][0] = 4 #The result is 0

shared_x = theano.shared(np.asarray(test_data[0], dtype=theano.config.floatX), borrow=True)
shared_y = theano.shared(np.asarray(test_data[1], dtype=theano.config.floatX), borrow=True)
test_data_shared = shared_x, theano.tensor.cast(shared_y, "int32")

gg = open('savedSelf.p', 'rb')
savedNet = cPickle.load(gg)
gg.close()
out_1 = savedNet.singleOutputPrediction(test_data_shared, 0)
print "The result is %s" % out_1