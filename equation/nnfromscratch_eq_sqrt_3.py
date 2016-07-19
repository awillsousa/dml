import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

fact = 10
nr_examples= 50
modelfilename = '../data/nnfromscratch_eq_3' + str(nr_examples) + '.pkl'

def activation(x):
    return np.tanh(x) #1/ (1 + np.exp(-x)) -0.5 #np.tanh(x) #  np.log(1 + np.exp(x))

def the_function(x):
    return  np.sqrt(np.abs(x))

test_nr = 10
#inputs_test = 12 * np.random.random((test_nr, nn_input_dim)) - 6
inputs_test = fact * (np.arange(1, test_nr + 1)).reshape(test_nr,1)

nn_input_dim = 1 # input layer dimensionality
inputs = fact * test_nr * (np.random.random((nr_examples, nn_input_dim)))
targets = the_function(inputs)

y = targets[...,0].reshape(nr_examples,1)

num_examples = len(targets) # training set size

nn_output_dim = 1 # output layer dimensionality

# Gradient descent parameters (I picked these by hand)
reg_lambda = 0.0 # 0.01 # regularization strength

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X = inputs):
    W1, b1, W2, b2, w3, b3 , w4, b4= model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = activation(z1)
    z2 = a1.dot(W2) + b2
    a2 = activation(z2)
    z3 = a2.dot(w3) + b3
    a3 = activation(z3)
    z4 = a3.dot(w4) + b4

    #exp_scores = np.exp(z2)
    #probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    #corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sqrt(np.sum((z4-y)**2))
    #data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    #data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
def predict(model, x=inputs, res = y):
    W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = activation(z1)
    z2 = a1.dot(W2) + b2
    a2 = activation(z2)
    z3 = a2.dot(W3) + b3
    a3 = activation(z3)
    z4 = a3.dot(W4) + b4
    #exp_scores = np.exp(z2)
    #probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    print("mean error " + str(np.mean(np.abs(z4-res))))
    return z4


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, X = inputs, num_passes=50000, print_loss=False,save =True , epsilon = 0.001):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.ones((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim)
    b2 = np.ones((1, nn_hdim))
    W3 = np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim)
    b3 = np.ones((1, nn_hdim))
    W4 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b4 = np.ones((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    old_loss = 1000.
    # Gradient descent. For each batch...
    for i in xrange(0, num_passes + 1):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = activation(z1)
        z2 = a1.dot(W2) + b2
        a2 = activation(z2)
        z3 = a2.dot(W3) + b3
        a3 = activation(z3)
        z4 = a3.dot(W4) + b4
        #exp_scores = np.exp(z2)
        #probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta5 = z4-y
        dW4 = (a3.T).dot(delta5)
        db4 = np.sum(delta5, axis=0, keepdims=True)
        delta4 = delta5.dot(W4.T) * (1 - np.power(a3, 2))
        dW3 = (a2.T).dot(delta4)
        db3 = np.sum(delta4, axis=0, keepdims=True)
        delta3 = delta4.dot(W3.T) * (1 - np.power(a2, 2))
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        #print('delta2 = ' + str(delta2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW4 += reg_lambda * W4
        dW3 += reg_lambda * W3
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        W3 += -epsilon * dW3
        b3 += -epsilon * db3
        W4 += -epsilon * dW4
        b4 += -epsilon * db4

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 , 'W3': W3, 'b3': b3, 'W4': W4, 'b4': b4}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        loss = calculate_loss(model)
        if print_loss and i % 10000 == 0:
            print 'Loss after iteration ' +str(i) + ': ' + str(loss)
            if loss > (old_loss):
                print(str(i)+' loss = ' + str(loss) +' old_loss = ' + str(old_loss))
                break
            old_loss = np.min((loss, old_loss))

        if save:
            gg = open(modelfilename, 'wb')
            pickle.dump(model, gg, protocol=pickle.HIGHEST_PROTOCOL)
            gg.close()
    return model

def load_params(filename):
    gg = open(filename, 'rb')
    params = pickle.load(gg)
    gg.close()
    return params

# Build a model with a 3-dimensional hidden layer
model = build_model(5, print_loss=True)

model = load_params(modelfilename)


targets = the_function(inputs)
y = targets
targets_test = the_function(inputs_test)
y_test = targets_test.reshape(test_nr,1)
pred = predict(model,  x=inputs_test, res = y_test)
sortx = np.sort(inputs_test[...,0])
sortedarrA = np.asarray(sorted(zip(inputs_test, pred[...,0])))
sortedarrB = np.asarray(sorted(zip(inputs_test,targets_test[...,0])))
predicplot, = plt.plot(sortx, sortedarrA[...,1],  marker='o', linestyle='--', color='r',  label='Prediction')
actualplot, = plt.plot(sortx, sortedarrB[...,1], label='Actual')
plt.legend(handles=[predicplot, actualplot])
plt.title('Three hidden layers')
plt.show()

#print(pred)


