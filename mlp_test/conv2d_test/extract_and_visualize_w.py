import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

filename = '../../data/models/best_model_convolutional_mlp_700_zero_angles_10_-10_.pkl'

gg = open(filename, 'rb')
params = pickle.load(gg)
gg.close()

W_1 = params[0].eval()
W_2 = params[2].eval()

for i in range(50):
        plt.subplot(5, 10, i + 1)
        fig = plt.imshow(params[4].eval()[i,1], cmap='Greys', interpolation='nearest')
        plt.title(str("conv 1 - " + str(i)), fontsize=7)
        fig.axes.get_xaxis().set_ticks([])
        fig.axes.get_yaxis().set_ticks([])


plt.show()


