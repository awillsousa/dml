# modified from https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/mlp.py
# (c) 2010--2015, Deep Learning Tutorials Development Team
# added functionality for saving the parameteres, reloading and testing the model.

"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import gzip

import numpy

import fli

import theano
import theano.tensor as T
import cPickle as pickle

import logging

from data_utils import get_blurred_sets, get_rotated_sets,shuffle_in_unison, save_model, load_params, epoch_from_filename

add_blurs = False
testrun = False
randomInit = False

logfilename= '../logs/mlp_modified.log'

activation_mlp=T.tanh
n_epochs_mlp=1000
saveepochs_mlp = numpy.arange(0, n_epochs_mlp + 1, 10)
loadparams = False
#If loadparams is True, then the parameters are loaded from this file,
# n_epochs_mlp must be greater than the starting epoch number,
# which is extracted from the paramsfilename.
paramsfilename = '../data/models/best_model_mlp_500_zero.pkl'


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, randomInit=False, loadparams=False, paramsfilename = paramsfilename ):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        loadedparams =[None]*4
        if loadparams:
            print("Loading params from "  + paramsfilename + "..." )
            loadedparams = load_params(paramsfilename)

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            W=loadedparams[0],
            b= loadedparams[1],
            activation=activation_mlp
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            randomInit=randomInit,
            W=loadedparams[2],
            b=loadedparams[3],
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input

# def __getstate__(self): return self.__dict__
# def __setstate__(self, d): self.__dict__.update(d)

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=activation_mlp):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, randomInit=False, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            initW = numpy.zeros(
                        (n_in, n_out),
                        dtype=theano.config.floatX
                    )
            initB = numpy.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                )
            if randomInit:
                initW[:] = numpy.random.randn(*initW.shape)
                initB[:] = numpy.random.randn(*initB.shape)
            self.W = theano.shared(
                value= initW,
                name='W',
                borrow=True
            )
            # initialize the biases b as a vector of n_out 0s
            self.b = theano.shared(
                value=initB,
                name='b',
                borrow=True
            )
        else:
            self.W = W
            self.b = b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset, add_the_blurs=False, blur=1, replace_images = False, angles = []):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    naughty_images = [132, 494, 902, 2720, 4476, 6885, 10994, 11949, 19360, 21601, 25159,
                      25562, 25678, 26504, 26560, 31596, 34404, 35234, 35480, 35616,
                      36104, 37038, 37816, 37834, 38526, 38700, 42566, 43109, 43454,
                      45143, 46078, 47034, 47600]

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    if add_the_blurs:
        blur_set = get_blurred_sets(train_set[0], train_set[1], blur)
        train_set = shuffle_in_unison(numpy.concatenate((train_set[0], blur_set[0])), numpy.concatenate((train_set[1], blur_set[1])))
    the_set =  train_set
    for the_angle in angles:
        rotated_set = get_rotated_sets(train_set[0], train_set[1], the_angle)
        the_set = numpy.concatenate((the_set[0], rotated_set[0])), numpy.concatenate((the_set[1], rotated_set[1]))
    train_set = shuffle_in_unison(the_set[0], the_set[1])
    if replace_images:
        test_set_x, test_set_y = train_set
        j = 0
        for i in naughty_images:
            test_set_x[i] = test_set_x[j]
            test_set_y[i] = test_set_y[j]
            j += 1
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def predict_mlp_all_fast(filename, test_data='test', saveToFile=False, diagnose=False):
    gg = open(filename, 'rb')
    params = pickle.load(gg)
    gg.close()

    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    if(test_data=='test'):
        test_set_x, test_set_y = datasets[2]
        test_data_str = '_test'
    elif (test_data == 'validation'):
        test_set_x, test_set_y = datasets[1]
        test_data_str = '_validation'
    elif (test_data=='train'):
        test_set_x, test_set_y = datasets[0]
        test_data_str = '_train'

    index = T.lscalar()
    hidden_output = activation_mlp(T.dot(test_set_x[index], params[0]) + params[1])
    final_output = T.dot(hidden_output, params[2]) + params[3]
    p_y_given_x = T.nnet.softmax(final_output)
    y_pred = T.argmax(p_y_given_x, axis=1)
    ind_arr = numpy.arange(10, dtype=numpy.uint8)
    infofunc = theano.function([index], p_y_given_x)
    testfunc = theano.function([index], [y_pred[0], test_set_y[index]])
    range= test_set_x.shape[0].eval()
    wrongpredictions = []
    for j in xrange (range):
        pred = testfunc(j)
        if not(pred[0] == pred[1]):
            print ("The predicted value " + str(pred[0]) + " at index " + str(j) + " is wrong. The correct value is " + str(pred[1]) +".")
            wrongpredictions.append([j,int(pred[0]),int(pred[1]), test_set_x[j]])
            if diagnose:
                err_arr = sorted(zip(ind_arr,infofunc(j)[0]), key=lambda x: x[1], reverse=True)
                print(err_arr)
                print('---')
    print ('There are ' + str(len(wrongpredictions)) + ' errors.')
    if saveToFile:
        gg = open('../data/mlp_test_errors'+test_data_str+'.pkl', 'wb')
        pickle.dump(wrongpredictions, gg, protocol=pickle.HIGHEST_PROTOCOL)
        gg.close()
    return wrongpredictions



def testfunction(i, params, test_set_x, test_set_y):
    index = T.lscalar()
    x = T.matrix('x')
    hidden_output = activation_mlp(T.dot(test_set_x[index], params[0]) + params[1])
    final_output = T.dot(hidden_output, params[2]) + params[3]
    p_y_given_x = T.nnet.softmax(final_output)
    y_pred = T.argmax(p_y_given_x, axis=1)
    testfunc = theano.function([index], [y_pred[0], test_set_y[index]])
    return testfunc(i)

def load_and_predict_custom_image(modelFilename, testImgFilename, testImgvalue, testImgFilenameDir='../data/custom/'):
    gg = open(modelFilename, 'rb')
    params = pickle.load(gg)
    gg.close()

    test_img = fli.processImg(testImgFilenameDir, testImgFilename)
    hidden_output = activation_mlp(T.dot(test_img, params[0]) + params[1])
    final_output = T.dot(hidden_output, params[2]) + params[3]
    p_y_given_x = T.nnet.softmax(final_output)
    y_pred = T.argmax(p_y_given_x, axis=1)
    testfunc = theano.function([], [y_pred[0]])
    prediction = testfunc()[0]
    correct = (testImgvalue == prediction)
    print('The prediction ' + str(testfunc()[0]) + ' for ' + testImgFilename + '  is ' + str(correct) + '.')
    return correct

def predict_mlp(filename, i, test_train_data = False):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    gg = open(filename, 'rb')
    params = pickle.load(gg)
    gg.close()

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    if (test_train_data):
        test_set_x, test_set_y = datasets[0]
    else:
        test_set_x, test_set_y = datasets[2]

    return testfunction(i, params, test_set_x, test_set_y), test_set_x[i], test_set_y[i]


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=n_epochs_mlp,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500, randomInit=randomInit,
             logfilename=logfilename, loadparams = loadparams, paramsfilename=paramsfilename,
             testrun = testrun, add_blurs = add_blurs):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """

    print ('loadparams is ' + str(loadparams))
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    randgen = 1234
    if randomInit:
        randgen = 3421
    rng = numpy.random.RandomState(randgen)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        randomInit=randomInit,
        loadparams = loadparams,
        paramsfilename = paramsfilename
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    # CCC Commenting out patience for simplicity and transparency's sake
    # patience = 10000  # look as this many examples regardless
    # patience_increase = 2  # wait this much longer when a new best is
    #                        # found
    # improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = n_train_batches # CCCmin(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    if loadparams:
        epoch = epoch_from_filename(paramsfilename)
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )


                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    # CCC if (
                    #     this_validation_loss < best_validation_loss *
                    #     improvement_threshold
                    # ):
                    #     patience = max(patience, iter *patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            # CCC if patience <= iter:
            #     done_looping = True
            #     break

        if epoch in saveepochs_mlp:
            # test it on the test set
            epoch_test_losses = [test_model(i) for i
                                 in range(n_test_batches)]
            epoch_test_score = numpy.mean(epoch_test_losses)
            print(('epoch %i, test error of '
                   'best model %f %%') %
                  (epoch, epoch_test_score * 100.))
            save_model(classifier.params, epoch, best_validation_loss, epoch_test_score,
                       '../data/models/best_model_mlp_'
                       , randomInit, add_blurs, testrun, logfilename, endrun = (n_epochs==epoch))

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    # test it on the test set
    final_test_losses = [test_model(i) for i
                   in range(n_test_batches)]
    final_test_score = numpy.mean(final_test_losses)
    print(('The final test score is %f %% ') %
          (final_test_score * 100.))

    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)



if __name__ == '__main__':
    logging.basicConfig(filename=logfilename, level=logging.INFO)
    test_mlp()

