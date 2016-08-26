import gzip
import pickle
import numpy as np
import sys
sys.path.insert(0, '../mlp_test')
from  data_utils import load_mnist

train_set, valid_set, test_set = load_mnist()

train_x, train_y = train_set

import matplotlib.cm as cm
import matplotlib.pyplot as plt
# plt.imshow(train_x[0].reshape((28, 28)), cmap = cm.Greys_r)
# plt.show()

#example_index=np.random.randint(10000)
example_index=9904
test_x, test_y = test_set
plt.title(test_y[example_index],  fontsize=24)
plt.imshow(test_x[example_index].reshape((28, 28)), cmap = cm.Greys_r)
plt.show()
