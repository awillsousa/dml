import cPickle

chosen_number = 7
filepath = "../data/savedImage_" + str(chosen_number) + ".p"

test_saved = cPickle.load( open(filepath, "rb" ) )

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