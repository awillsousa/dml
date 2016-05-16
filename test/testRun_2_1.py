import numpy as np
import network2
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.load('net2')
imgIndex = 45
testImg = mnist_loader.load_data_wrapper()[2][imgIndex]

prediction = np.argmax(net.feedforward(testImg[0]))
value = testImg[1]

print "The predicted value is " + str(prediction) +" . The actual value is " + str(value) +"."

