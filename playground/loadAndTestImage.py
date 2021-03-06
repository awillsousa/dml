import gzip
import theano
import cPickle
import numpy as np
import network3_2

chosen_number = 1
filepath = "../data/savedImage_" + str(chosen_number) + ".p"

test_saved = cPickle.load( open(filepath, "rb" ) )
test_x_saved, test_y_saved = test_saved

import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist


test_data = load_mnist[2]


test_data[0][0] = test_x_saved
test_data[1][0] = test_y_saved #The result is 1

#test_data[1][0] = 4 #The result is 0

shared_x = theano.shared(np.asarray(test_data[0], dtype=theano.config.floatX), borrow=True)
shared_y = theano.shared(np.asarray(test_data[1], dtype=theano.config.floatX), borrow=True)
test_data_shared = shared_x, theano.tensor.cast(shared_y, "int32")

gg = open('savedSelf.p', 'rb')
savedNet = cPickle.load(gg)
gg.close()
out_1 = savedNet.singleOutputPrediction(test_data_shared, chosen_number)
result = "correct" if out_1 == 1 else "Wrong"
print "The saved number is " + str(test_y_saved) + ". The prediction is " + str(chosen_number) + " and hence %s" % result

