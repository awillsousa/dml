import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy

rng = numpy.random.RandomState(23450)

# instantiate 4D tensor for input
input = T.tensor4(name='input')

nr_channels = 5
w_bound = numpy.sqrt(4 * 9 * 9)

def func(i, j, k):
    if k == 0:
        if (i == 4 or j==4):
            return 0.
        else:
            return 1./w_bound
    elif k == 1:
        if (i == j):
            return 0.
        else:
            return 1./w_bound
    elif k == 2:
        if (i == 8 - j):
            return 0
        else:
            return 1. / w_bound
    elif k == 3:
        if (i == 2*j or j == i*2):
            return 0
        else:
            return 1. / w_bound



# initialize shared variable for weights.
w_shp = (nr_channels, 4, 9, 9)
at = numpy.zeros(w_shp)
b =[]

for m in range(4):
    a = numpy.frompyfunc(
        lambda i, j: func(i, j, 3 -m), 2, 1
    ).outer(
        numpy.arange(9),
        numpy.arange(9),
    ).astype(numpy.float64)  # a_ij = func(i, j)
    b.append(a)
at[:,]=b
W = theano.shared( at, name ='W')

# initialize shared variable for bias (1D tensor) with random values
# IMPORTANT: biases are usually initialized to zero. However in this
# particular application, we simply apply the convolutional layer to
# an image without learning the parameters. We therefore initialize
# them to random values to "simulate" learning.
b_shp = (nr_channels,)
b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input.dtype), name ='b')

# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv2d(input, W)

# build symbolic expression to add bias and apply activation function, i.e. produce neural net layer output
# A few words on ``dimshuffle`` :
#   ``dimshuffle`` is a powerful tool in reshaping a tensor;
#   what it allows you to do is to shuffle dimension around
#   but also to insert new ones along which the tensor will be
#   broadcastable;
#   dimshuffle('x', 2, 'x', 0, 1)
#   This will work on 3d tensors with no broadcastable
#   dimensions. The first dimension will be broadcastable,
#   then we will have the third dimension of the input tensor as
#   the second of the resulting tensor, etc. If the tensor has
#   shape (20, 30, 40), the resulting tensor will have dimensions
#   (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor)
#   More examples:
#    dimshuffle('x') -> make a 0d (scalar) into a 1d vector
#    dimshuffle(0, 1) -> identity
#    dimshuffle(1, 0) -> inverts the first and second dimensions
#    dimshuffle('x', 0) -> make a row out of a 1d vector (N to 1xN)
#    dimshuffle(0, 'x') -> make a column out of a 1d vector (N to Nx1)
#    dimshuffle(2, 0, 1) -> AxBxC to CxAxB
#    dimshuffle(0, 'x', 1) -> AxB to Ax1xB
#    dimshuffle(1, 'x', 0) -> AxB to Bx1xA


output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# create theano function to compute filtered images
f = theano.function([input], output)

from PIL import Image


# open random image of dimensions 639x516
img = Image.open(open('../../data/pics/venus.png'))
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float64') / 256.



# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 4, img.shape[0], img.shape[1])
filtered_img = f(img_)


import pylab


# plot original image and first and second components of output
pylab.subplot((nr_channels + 1)/2, 2 , 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
for i in range(nr_channels):
    pylab.subplot((nr_channels + 1)/2, 2 , i +2);
    pylab.axis('off'); pylab.imshow(filtered_img[0, i, :, :])
pylab.show()