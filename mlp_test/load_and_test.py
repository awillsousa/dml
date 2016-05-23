import theano
import cPickle as pickle

import theano.tensor as T

from mlp_modified import  load_data

dataset='mnist.pkl.gz'

datasets = load_data(dataset)

test_set_x, test_set_y = datasets[2]

gg = open('best_model_mlp_100.pkl', 'rb')
params= pickle.load(gg)
gg.close()

index = T.lscalar()
x = T.matrix('x')
hidden_output = T.dot(test_set_x[index], params[0]) + params[1]
final_output = T.dot(hidden_output, params[2]) + params[3]
p_y_given_x = T.nnet.softmax(final_output)
y_pred = T.argmax(p_y_given_x, axis=1)
y = T.ivector('y')
testfunc = theano.function([index], [y_pred, test_set_y[index]])
print(testfunc(1)[0])
# output = theano.function(inputs=[index], outputs=y_pred,
#             givens={x: test_set_x[index * batch_size:(index + 1) * batch_size],
#             y: test_set_y[index * batch_size:(index + 1) * batch_size]
#         })



# print(str(output(1)))