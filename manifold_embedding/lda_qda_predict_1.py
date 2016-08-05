from __future__ import division

import numpy as np
import gzip
import cPickle as pickle
from sklearn import discriminant_analysis



showAll = True
plotVertexImages = False
selected_length_train = 50000
selected_start_train=0
selected_length_test = 10000
selected_start_test=0

target_values = np.array([ 2, 3, 5])

filename = "../data/mnist.pkl.gz"
f = gzip.open(filename, 'rb')
train_data, verificatio_data, test_data = pickle.load(f)
f.close()

selected_train = [index for index in range(selected_start_train, selected_start_train + selected_length_train) if train_data[1][index] in target_values]
selected_test = [index for index in range(selected_start_test, selected_start_test + selected_length_test) if test_data[1][index] in target_values]

indexes = np.asarray([i for i in selected_train])
X_data_train = np.asarray([train_data[0][i] for i in selected_train])
y_data_train = np.asarray([train_data[1][i] for i in selected_train])
X_data_test = np.asarray([test_data[0][i] for i in selected_test])
y_data_test = np.asarray([test_data[1][i] for i in selected_test])


print("Computing Linear Discriminant Analysis projection")
X_train = X_data_train.copy()
X_train.flat[::X_data_train.shape[1] + 1] += 0.01  # Make X_data invertible
lda = discriminant_analysis.LinearDiscriminantAnalysis().fit(X_train, y_data_train)
Y_pred_test_lda = lda.predict(X_data_test)
num_errors_lda = 0
len_test = len(Y_pred_test_lda)
for i in range(0, len_test):
    if (Y_pred_test_lda[i] != y_data_test[i]):
        num_errors_lda += 1
        #print('Wrong prediction ' + str(Y_pred_test[i]) + ' != ' + str(y_data_test[i]))
print('LDA ' + str(num_errors_lda) + ' wrong predictions out of ' + str(len_test) )
print ('LDA ' +"{:4.2f}".format(num_errors_lda * 100 / len_test) + '% error rate ')

qda = discriminant_analysis.QuadraticDiscriminantAnalysis().fit(X_train, y_data_train)
num_errors_qda = 0
Y_pred_test_qda = qda.predict(X_data_test)
for i in range(0, len_test):
    if (Y_pred_test_qda[i] != y_data_test[i]):
        num_errors_qda += 1
        #print('Wrong prediction ' + str(Y_pred_test[i]) + ' != ' + str(y_data_test[i]))
print('QDA ' + str(num_errors_qda) + ' wrong predictions out of ' + str(len_test))
print ('QDA ' +"{:4.2f}".format(num_errors_qda * 100 / len_test) + '% error rate ')


