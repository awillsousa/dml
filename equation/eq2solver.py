import numpy as np
import numpy.testing as npt

def activation( x, deriv = False):
    if deriv == True:
        return  x * (1 - x)
    return 50/( 1 + np.exp(-x))

traindim= 5
x = np.random.random((traindim, 1))
y = x**2


np.random.seed(1)

compdim = 1

syn0 = 2 * np.random.random((compdim,traindim)) -1
syn1 = 2 * np.random.random((traindim,1)) -1

old_error = 0.0

for j in xrange(100000):

    l0 = x
    l1 = activation(l0.dot(syn0))
    #print('max syn0 = ' +str(np.max(syn0)))
    l2 = activation(l1.dot(syn1))
    #print('max l2 = ' +str(np.max(l2)))

    l2_error = y -l2
    #print('max error = '+str(np.max(abs(l2_error)))+ ' min error = ' +str(np.min(abs(l2_error))))
    if j % 10000 == 0:
        print(str(j) + ' - Error: ' + str(np.mean(np.abs(l2_error))))
        print(str(j) + ' - Diff Error: ' + str(np.mean(np.abs(l2_error - old_error))))

    old_error= l2_error

    l2_delta = l2_error * activation(l2, deriv = True)

    l1_error = l2_delta.dot(syn1.T)
    #print(np.max(l2_delta))
    l1_delta = l1_error * activation(l1, deriv=True)
    #print(np.max(l1_error))
    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)
