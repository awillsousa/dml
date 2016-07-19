import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([0, 1, 2, 3, 4, 5]).reshape(6,1) #training data X
len_inp = inputs.shape[0]
exp_y = np.array([1.0, 0.5, 1, 1.5, 1.0, 1.5])

x = T.dvector()
y = T.dscalar()

np.random.seed(6) #6 is good

def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x) #theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
    h = nnet.relu(m)
    return h

def grad_desc(cost, theta):
    alpha = 0.01 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))


theta1 = theano.shared(np.array(np.random.rand(2,len_inp -1), dtype=theano.config.floatX)) # randomly initialize
theta2 = theano.shared(np.array(np.random.rand(len_inp,1), dtype=theano.config.floatX))

hid1 = layer(x, theta1) #hidden layer

out1 = T.sum(layer(hid1, theta2)) #output layer
fc = (out1 - y)**2 #cost expression

cost = theano.function(inputs=[x, y], outputs=fc, updates=[
    (theta1, grad_desc(fc, theta1)),
    (theta2, grad_desc(fc, theta2))])

cur_cost = 0
for i in range(30000):
    for k in range(len(inputs)):
        cur_cost = cost(inputs[k], exp_y[k]) #call our Theano-compiled cost function, it will auto update weights
    if i % 500 == 0: #only print the cost every 500 epochs/iterations (to save space)
        print('Cost: %s' % (cur_cost,))

run_forward = theano.function(inputs=[x], outputs=out1)

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