import cPickle as pickle
from os import listdir
from os.path import isfile, join
from SdA_modified import predict_all_mnist_test_images_sda,predict_custom_image_sda

paramsFilename = 'best_model_sda_5_50.pkl'

#Test 1
gg = open(paramsFilename, 'rb')
params = pickle.load(gg)
gg.close()

path = '../data/custom'

files = [f for f in listdir(path) if isfile(join(path, f))]
for file in files:
    test_img_value = filter(str.isdigit, file)
    predict_custom_image_sda(params,file)

#Test_2
predict_all_mnist_test_images_sda(paramsFilename)

