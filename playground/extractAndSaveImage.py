import gzip
import cPickle
import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist


train_set, valid_set, test_set = load_mnist()


chosen_index = 1250

test_x_chosen = test_set[0][chosen_index]
test_y_chosen = test_set[1][chosen_index]

test_chosen = [test_set[0][chosen_index], test_set[1][chosen_index]]

filepath = "../data/savedImage_" + str(test_y_chosen) + ".p"

cPickle.dump(test_chosen, open(filepath, "wb" ) )

test_saved = cPickle.load( open(filepath, "rb" ) )

test_x_saved, test_y_saved = test_saved

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.title(test_y_saved,  fontsize=24)
plt.imshow(test_x_saved.reshape((28, 28)), cmap = cm.Greys_r)
plt.show()
