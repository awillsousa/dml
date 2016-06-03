from  mlp_modified import predict_mlp,  predict_mlp_all_fast, load_and_predict_custom_image

#Uncomment to test

filename = 'best_model_mlp_500.pkl'

# Test 1
index = 9
pred = predict_mlp(filename, index)
correct = (pred[0] == pred[1])
print('The prediction for index '+str(index) + ' is ' + str(pred[0])+'.')
print ('The correct value is ' + str(pred[1])+'.')
print ('The prediction is ' + str(correct)+'.')

# Test 2
from os import listdir
from os.path import isfile, join
path = '../data/custom'
files = [f for f in listdir(path) if isfile(join(path, f))]
for file in files:
    test_img_value = filter(str.isdigit, file)
    load_and_predict_custom_image(filename,file, int(test_img_value))


# Test 3
predict_mlp_all_fast(filename)