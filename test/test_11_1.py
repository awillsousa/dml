import numpy as np

def nonlinear( x, deriv = False):
    if deriv == True:
        return  x * (1 - x)
    return 1/( 1 + np.exp(-x))

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

syn0 = 2 * np.random.random((3,4)) -1
syn1 = 2 * np.random.random((4,1)) -1

for j in xrange(60000):

    l0 = X
    l1 = nonlinear(l0.dot(syn0))
    l2 = nonlinear(l1.dot(syn1))

    l2_error = y -l2

    if j % 10000 == 0:
        print('Error: ' + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * nonlinear(l2, deriv = True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlinear(l1, deriv=True)

    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)
