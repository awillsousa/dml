import cPickle as pickle
import convolutional_mlp_modified as cmlp

paramsFilename = 'best_model_convolutional_mlp_100.pkl'

# Test 1
from os import listdir
from os.path import isfile, join

gg = open(paramsFilename, 'rb')
params = pickle.load(gg)
gg.close()
path = '../data/custom'
files = [f for f in listdir(path) if isfile(join(path, f))]
for file in files:
    test_img_value = filter(str.isdigit, file)
    cmlp.predict_custom_image(params,file)

# Test 2
cmlp.predict_all_mnist_test_images(paramsFilename)
