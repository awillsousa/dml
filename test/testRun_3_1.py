import network3
test_data = network3.load_data_shared()[2]
import cPickle

gg = open('net3.p', 'rb')
net3= cPickle.load(gg)
gg.close()

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

imgIndex = 45
testImg = mnist_loader.load_data_wrapper()[2][imgIndex]

print str(net3.feedback2(testImg[0]))
