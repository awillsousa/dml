import cPickle as pickle
from os import listdir
from os.path import isfile, join
from SdA_modified import predict_mnist_sda,predict_custom_image_sda

paramsFilename = '../data/models/best_model_sda_3_50.pkl'

#Test 1
gg = open(paramsFilename, 'rb')
params = pickle.load(gg)
gg.close()

path = '../data/custom'

files = [f for f in listdir(path) if isfile(join(path, f))]
n_right = 0
n_tot = len(files)
for file in files:
    test_img_value = filter(str.isdigit, file)
    n_right += predict_custom_image_sda(params,file)
print(str(n_tot - n_right) + ' wrong predictions out of ' + str(n_tot))

#Test_2
#predict_mnist_sda(paramsFilename)

