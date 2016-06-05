import cPickle as pickle
from convolutional_mlp_modified import predict_all_mnist_test_images, predict_custom_image

paramsFilename = '../data/models/best_model_convolutional_mlp_250.pkl'

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
    predict_custom_image(params,file)

# Test 2
predict_all_mnist_test_images(paramsFilename)
