import gzip
import cPickle
import numpy as np
with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)

chosen_index = 0

test_x_chosen = test_set[0][chosen_index]
test_y_chosen = test_set[1][chosen_index]

test_chosen = [test_set[0][chosen_index], test_set[1][chosen_index]]

cPickle.dump(test_chosen, open( "savedImage.p", "wb" ) )

test_saved = cPickle.load( open("savedImage.p", "rb" ) )

test_x_saved, test_y_saved = test_saved

import matplotlib.cm as cm
import matplotlib.pyplot as plt
# plt.imshow(train_x[0].reshape((28, 28)), cmap = cm.Greys_r)
# plt.show()

#example_index=np.random.randint(10000)
#example_index=9904
#test_x, test_y = test_set
plt.title(test_y_saved,  fontsize=24)
plt.imshow(test_x_saved.reshape((28, 28)), cmap = cm.Greys_r)
plt.show()
