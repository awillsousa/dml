import convolutional_mlp_modified as cmlp

# Test 1
from os import listdir
from os.path import isfile, join
path = '../data/custom'
files = [f for f in listdir(path) if isfile(join(path, f))]
for file in files:
    test_img_value = filter(str.isdigit, file)
    cmlp.predict_custom_image('best_model_convolutional_mlp10.pkl',file)