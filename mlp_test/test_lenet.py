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
n_right = 0
n_tot = len(files)
for file in files:
    test_img_value = filter(str.isdigit, file)
    n_right += predict_custom_image(params,file)
print(str(n_tot - n_right) + ' wrong predictions out of ' + str(n_tot))

# Test 2
predict_all_mnist_test_images(paramsFilename)
