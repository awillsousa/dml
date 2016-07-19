import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32,exception_verbosity=high"
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4) #4 is good

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
theta2 = theano.shared(np.array(np.random.rand(5,4), dtype=theano.config.floatX))
theta3 = theano.shared(np.array(np.random.rand(5,1), dtype=theano.config.floatX))

hid1 = layer(x, theta1) #hidden layer
hid2 = layer(x, theta2) #hidden layer

out1 = T.sum(layer(layer(hid1, theta2), theta3)) #output layer
fc = (out1 - y)**2 #cost expression

cost = theano.function(inputs=[x, y], outputs=fc, updates=[
    (theta1, grad_desc(fc, theta1)),
    (theta2, grad_desc(fc, theta2)),
    (theta3, grad_desc(fc, theta3))])

inputs = np.array([0, 1, 2, 3, 4]).reshape(5,1) #training data X
exp_y = np.array([2, 1, 2, 0, 2]) #training data Y
cur_cost = 0
for i in range(10000):
    for k in range(len(inputs)):
        cur_cost = cost(inputs[k], exp_y[k]) #call our Theano-compiled cost function, it will auto update weights
    if i % 500 == 0: #only print the cost every 500 epochs/iterations (to save space)
        print('Cost: %s' % (cur_cost,))

run_forward = theano.function(inputs=[x], outputs=out1)

#Training done! Let's test it out

marg = 1
inf_ax = np.min(inputs)-marg
sup_ax = np.max(inputs)+marg
interval = np.linspace(inf_ax ,sup_ax ,100)
output=[]
for x in np.nditer(interval):
    output.append(run_forward([x]))

plt.axis([inf_ax, sup_ax, np.min(exp_y)-marg, np.max(exp_y) +marg])
plt.plot(interval, np.asarray(output))
plt.plot(inputs, exp_y ,  marker='o', color='r', linestyle='', label='Expected')
plt.show()


