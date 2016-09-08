# Modified from https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/convolutional_mlp.py
# (c) 2010--2015, Deep Learning Tutorials Development Team
# added functionality for saving the parameteres, reloading and testing the model on custom images.
"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

from __future__ import print_function

import os
import sys
import timeit

import numpy

import cPickle as pickle

from data_utils import save_model, load_params, epoch_from_filename

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from mlp_modified import HiddenLayer, LogisticRegression, load_data

import fli

import logging

logfilename= '../logs/mlp_convolutional_modified.log'

activation_convmlp= T.nnet.relu #T.tanh
n_epochs_convmlp=1000
saveepochs_convmlp = numpy.arange(0, n_epochs_convmlp + 1, 10)

add_blurs = False
blur = 2
testrun= False
loadparams = False
rotation_angles = [10, 5, -5, -10]
#If loadparams is True, then the parameters are loaded from this file,
# n_epochs_mlp must be greater than the starting epoch number,
# which is extracted from the paramsfilename.
paramsfilename = '../data/models/best_model_convolutional_mlp_110_pars__zero_angles_10_5_-5_-10_.pkl'


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), W=None, b=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input
        if W is None:
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                       numpy.prod(poolsize))
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.W=W
            self.b=b

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = activation_convmlp(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

def evaluate_lenet5(learning_rate=0.1, n_epochs=n_epochs_convmlp, dataset='mnist.pkl.gz', nkerns=[20, 50],
            batch_size=500, thislogfilename = logfilename,
            loadparams=loadparams, paramsfilename=paramsfilename,
            randomInit=False, testrun=testrun, add_blurs=add_blurs, blur=blur, rot_angles = rotation_angles, annotation =''):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    loadedparams = [None] * 8
    if loadparams:
        print("Loading params from " + paramsfilename + "...")
        loadedparams = load_params(paramsfilename)

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset, add_the_blurs=add_blurs, blur = blur, angles = rot_angles)
    if len(rot_angles)>0:
        annotation += '_angles_'
        for ang in rot_angles:
            annotation += str(ang)+'_'



    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2),
        W = loadedparams[6],
        b=loadedparams[7]
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2),
        W = loadedparams[4],
        b = loadedparams[5]
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=activation_convmlp,
        W=loadedparams[2],
        b=loadedparams[3]
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output,
                                n_in=500, n_out=10,
                                W=loadedparams[0],
                                b=loadedparams[1])

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

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
    #                                # considered significant
    validation_frequency = n_train_batches #min(n_train_batches, patience // 2)
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

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    # CCC if this_validation_loss < best_validation_loss *  \
                    #    improvement_threshold:
                    #     patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            # CCC if patience <= iter:
            #     done_looping = True
            #     break

        if epoch in saveepochs_convmlp:
            # test it on the test set
            epoch_test_losses = [test_model(i) for i
                                 in range(n_test_batches)]
            epoch_test_score = numpy.mean(epoch_test_losses)
            print(('epoch %i, test error of '
                   'best model %f %%') %
                  (epoch, epoch_test_score * 100.))
            save_model(params, epoch, best_validation_loss, epoch_test_score, '../data/models/best_model_convolutional_mlp_'
                       , randomInit, add_blurs, testrun, thislogfilename, endrun = (n_epochs == epoch), annotation = annotation)

    end_time = timeit.default_timer()

    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)

def predict_on_mnist(modelfilename, activation=activation_convmlp, test_data='test', saveToFile=False, diagnose = False):

    gg = open(modelfilename, 'rb')
    params = pickle.load(gg)
    gg.close()


    nkerns = [20, 50]
    batch_size = 1
    poolsize = (2, 2)

    dataset = 'mnist.pkl.gz'
    datasets = load_data(
        dataset)
    if (test_data == 'test'):
        test_set_x, test_set_y = datasets[2]
        test_data_str = '_test'
    elif (test_data == 'validation'):
        test_set_x, test_set_y = datasets[1]
        test_data_str = '_validation'
    elif (test_data == 'train'):
        test_set_x, test_set_y = datasets[0]
        test_data_str = '_train'

    index = T.lscalar()
    layer0_input = test_set_x[index].reshape((batch_size, 1, 28, 28))

    conv_out_0 = conv2d(
        input=layer0_input,
        filters=params[6],
        input_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5)
    )

    # downsample each feature map individually, using maxpooling
    pooled_out_0 = downsample.max_pool_2d(
        input=conv_out_0,
        ds=poolsize,
        ignore_border=True
    )

    output_0 = activation(pooled_out_0 + params[7].dimshuffle('x', 0, 'x', 'x'))

    conv_out_1 = conv2d(
        input=output_0,
        filters=params[4],
        input_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
    )

    # downsample each feature map individually, using maxpooling
    pooled_out_1 = downsample.max_pool_2d(
        input=conv_out_1,
        ds=poolsize,
        ignore_border=True
    )

    output_1 = activation(pooled_out_1 + params[5].dimshuffle('x', 0, 'x', 'x'))
    output_2 = activation(T.dot(output_1.flatten(2), params[2]) + params[3])

    final_output = T.dot(output_2, params[0]) + params[1]
    p_y_given_x = T.nnet.softmax(final_output)
    y_pred = T.argmax(p_y_given_x, axis=1)
    ind_arr = numpy.arange(10, dtype=numpy.uint8)
    testfunc = theano.function([index], [y_pred[0], test_set_y[index]])
    infofunc = theano.function([index], p_y_given_x)
    range = test_set_x.shape[0].eval()
    wrongpredictions = []
    for j in xrange(range):
        prediction = testfunc(j)
        correct = (prediction[0] == prediction[1])
        if correct == False:
            print('The prediction ' + str(prediction[0]) + ' for index ' + str(j) + '  is wrong . The correct value is '
                  + str(prediction[1]) + '.')
            if diagnose:
                err_arr = sorted(zip(ind_arr,infofunc(j)[0]), key=lambda x: x[1], reverse=True)
                print(err_arr)
                print('---')
            wrongpredictions.append([j, prediction[0], prediction[1], test_set_x[j]])
    print('There are ' + str(len(wrongpredictions)) + ' errors.')
    if saveToFile:
        gg = open('../data/lenet_test_errors' + test_data_str + '.pkl', 'wb')
        pickle.dump(wrongpredictions, gg, protocol=pickle.HIGHEST_PROTOCOL)
        gg.close()
    return wrongpredictions

def predict_custom_image(params, testImgFilename='own_0.png', activation= activation_convmlp, testImgFilenameDir = '../data/custom/'):

    test_img_value = filter(str.isdigit, testImgFilename)

    test_img = fli.processImg(testImgFilenameDir, testImgFilename)

    nkerns = [20, 50]
    batch_size = 1
    poolsize = (2, 2)

    layer0_input = test_img.reshape((batch_size, 1, 28, 28)).astype(numpy.float32)

    conv_out_0 = conv2d(
        input=layer0_input,
        filters=params[6],
        input_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5)
    )

    # downsample each feature map individually, using maxpooling
    pooled_out_0 = downsample.max_pool_2d(
        input=conv_out_0,
        ds=poolsize,
        ignore_border=True
    )

    output_0 = activation(pooled_out_0 + params[7].dimshuffle('x', 0, 'x', 'x'))

    conv_out_1 = conv2d(
        input=output_0,
        filters=params[4],
        input_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
    )

    # downsample each feature map individually, using maxpooling
    pooled_out_1 = downsample.max_pool_2d(
        input=conv_out_1,
        ds=poolsize,
        ignore_border=True
    )

    output_1 = activation(pooled_out_1 + params[5].dimshuffle('x', 0, 'x', 'x'))
    output_2 = activation(T.dot(output_1.flatten(2), params[2]) + params[3])

    final_output = T.dot(output_2, params[0]) + params[1]
    p_y_given_x = T.nnet.softmax(final_output)
    y_pred = T.argmax(p_y_given_x, axis=1)
    testfunc = theano.function([], [y_pred[0]])
    prediction = testfunc()[0]
    correct = (int(test_img_value) == prediction)
    print('The prediction ' + str(testfunc()[0]) + ' for ' + testImgFilename + '  is ' + str(correct) + '.')
    return correct


if __name__ == '__main__':
    logging.basicConfig(filename=logfilename, level=logging.INFO)
    evaluate_lenet5()


