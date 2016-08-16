import cPickle as pickle
from convolutional_mlp_modified import predict_on_mnist, predict_custom_image
from mlp_utils import isqrt

paramsFilename = '../data/models/best_model_convolutional_mlp_1000_zero.pkl'

# Test 1
from os import listdir
from os.path import isfile, join

def test_1(path = '../data/custom'):
    gg = open(paramsFilename, 'rb')
    params = pickle.load(gg)
    gg.close()
    files = [f for f in listdir(path) if isfile(join(path, f))]
    n_right = 0
    n_tot = len(files)
    for file in files:
        test_img_value = filter(str.isdigit, file)
        n_right += predict_custom_image(params,file, testImgFilenameDir = path)
    print(str(n_tot - n_right) + ' wrong predictions out of ' + str(n_tot))

test_1(path = '../data/pics/uic')

# Test 2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def test_2(filename, data_set='validation',  diagnosys=True, showImages= True):
    wrongpredictions = predict_on_mnist(filename, test_data=data_set, saveToFile=False,  diagnose=diagnosys)
    if showImages:
        i = 1
        a = min(10, isqrt(len(wrongpredictions)) + 1)
        for wimg in wrongpredictions:
            if ( i < 100):
                plt.subplot(a, a, i)
                plt.title(str(wimg[1])+'!='+ str(wimg[2]))
                fig = plt.imshow(wimg[3].reshape((28, 28)).eval(), cmap=cm.Greys_r)
                plt.ylabel(str(wimg[0]))
                fig.axes.get_xaxis().set_ticks([])
                fig.axes.get_yaxis().set_ticks([])
                i += 1
        plt.tight_layout()
        plt.show()

#test_2(filename  = paramsFilename )

#predict_on_mnist(paramsFilename, test_data='validation', saveToFile=False)
