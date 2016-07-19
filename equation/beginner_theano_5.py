import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
import matplotlib.pyplot as plt


x = T.dscalar()

x = T.dvector()
y = T.dscalar()

def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x) #theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
    h = 3 * nnet.sigmoid(m)
    return h

def grad_desc(cost, theta):
    alpha = 0.1 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))


theta1 = theano.shared(np.array(np.random.rand(2,4), dtype=theano.config.floatX)) # randomly initialize
theta2 = theano.shared(np.array(np.random.rand(5,1), dtype=theano.config.floatX))

hid1 = layer(x, theta1) #hidden layer

out1 = T.sum(layer(hid1, theta2)) #output layer
fc = (out1 - y)**2 #cost expression

cost = theano.function(inputs=[x, y], outputs=fc, updates=[
    (theta1, grad_desc(fc, theta1)),
    (theta2, grad_desc(fc, theta2))])

inputs = np.array([0, 1, 2, 3, 4]).reshape(5,1) #training data X
exp_y = np.array([0, 1, 2, 0, 2]) #training data Y
cur_cost = 0
for i in range(10000):
    for k in range(len(inputs)):
        cur_cost = cost(inputs[k], exp_y[k]) #call our Theano-compiled cost function, it will auto update weights
    if i % 500 == 0: #only print the cost every 500 epochs/iterations (to save space)
        print('Cost: %s' % (cur_cost,))

run_forward = theano.function(inputs=[x], outputs=out1)

#Training done! Let's test it out
print(np.floor(run_forward([0])+0.5))
print(np.floor(run_forward([1])+0.5))
print(np.floor(run_forward([2])+0.5))
print(np.floor(run_forward([3])+0.5))

print(run_forward([0]))
print(run_forward([1]))
print(run_forward([2]))
print(run_forward([3]))

interval = np.linspace(np.min(inputs)-0.5, np.max(inputs)+0.5,100)
output=[]
for x in np.nditer(interval):
    output.append(run_forward([x]))

plt.axis([np.min(inputs)-0.5, np.max(inputs)+0.5, np.min(exp_y)-0.5, np.max(exp_y) +0.5])
plt.plot(interval, np.asarray(output))
plt.plot(inputs, exp_y ,  marker='o', linestyle='', color='r',  label='Expected')
plt.show()
