import cPickle as pickle
from convolutional_mlp_modified import predict_on_mnist, predict_custom_image

paramsFilename = '../data/models/best_model_convolutional_mlp_1000_zero.pkl'

# Test 1
from os import listdir
from os.path import isfile, join

def test_1():
    gg = open(paramsFilename, 'rb')
    params = pickle.load(gg)
    gg.close()
    path = '../data/custom'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    n_right = 0
    n_tot = len(files)
    for file in files:
        test_img_value = filter(str.isdigit, file)
        n_right += predict_custom_image(params,file)
    print(str(n_tot - n_right) + ' wrong predictions out of ' + str(n_tot))


test_1()

# Test 2
#predict_on_mnist(paramsFilename, test_data='validation', saveToFile=False)

