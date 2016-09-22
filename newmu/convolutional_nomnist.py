# based on https://github.com/Newmu/Theano-Tutorials/blob/master/5_convolutional_net.py
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist, nomnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from data_utils import save_model, load_params
import scipy.ndimage.interpolation as scipint

noMNIST = False
loadparams = False
epoch = 0
paramsfilename = 'models/conv_net_7_pars__nomnist_-1.pkl'
rotation_angles = [] #[10, -5, 5 -10]
annotation = ''

srng = RandomStreams()

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def get_rotated_vector(vector, angle=0):
    return scipint.rotate(vector.reshape(28, 28), angle, order=0, reshape=False).flatten()

def epoch_from_filename(filename):
    return int(filter(str.isdigit, filename.split("pars")[0]))

def get_rotated_sets(set_x, set_y, angle=0):
    return np.apply_along_axis(get_rotated_vector, axis=1, arr=set_x, angle=angle), set_y

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2), ignore_border = False)
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2), ignore_border = False)
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2), ignore_border = False)
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

maxrange = -1
if noMNIST:
    annotation += '_nomnist_'+str(maxrange)
    trX, trY, vaX, vaY, teX, teY = nomnist(range = -1)
else:
    annotation += '_nomnist_'
    trX, teX, trY, teY = mnist(onehot=True)
if len(rotation_angles) > 0:
    annotation += '_angles_'
    the_set = trX, trY
    for ang in rotation_angles:
        annotation += str(ang) + '_'
        rotated_set = get_rotated_sets(trX, trY, ang)
        the_set = np.concatenate((the_set[0], rotated_set[0])), np.concatenate((the_set[1], rotated_set[1]))
    trX, trY = shuffle_in_unison(the_set[0], the_set[1])

print("noMNIST loaded" if noMNIST else "MNIST loaded")

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

X = T.ftensor4()
Y = T.fmatrix()

if loadparams:
    epoch = epoch_from_filename(paramsfilename)
    w, w2, w3, w4, w_o = load_params(paramsfilename)
else:
    w = init_weights((32, 1, 3, 3))
    w2 = init_weights((64, 32, 3, 3))
    w3 = init_weights((128, 64, 3, 3))
    w4 = init_weights((128 * 3 * 3, 625))
    w_o = init_weights((625, 10))


noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

def build():
    print('Starting ...')
    for i in range(epoch, 100):
        print('Working on epoch ' + str(i+1)+' ...')
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
                train(trX[start:end], trY[start:end])
        test_score = np.mean(np.argmax(teY, axis=1) == predict(teX))
        if (i+1) % 1 == 0:
            save_model(params, epoch=i+1, annotation=annotation, namestub='conv_net', test_score=test_score)

def test_prediction():
    test_score = np.mean(np.argmax(teY, axis=1) == predict(teX))
    print(('The model params saved in ' + paramsfilename
           + ' yield a test score of %f %%') % (test_score * 100.))

#test_prediction()
build()