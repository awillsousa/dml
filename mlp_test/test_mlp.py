from  mlp_modified import predict_mlp,  predict_mlp_all_fast, load_and_predict_custom_image
from mlp_utils import isqrt


filename = '../data/models/best_model_mlp_500_zero.pkl'

# Test 1
def test_1():
    index = 9
    pred = predict_mlp(filename, index)
    correct = (pred[0] == pred[1])
    print('The prediction for index '+str(index) + ' is ' + str(pred[0])+'.')
    print ('The correct value is ' + str(pred[1])+'.')
    print ('The prediction is ' + str(correct)+'.')

# Test 2
from os import listdir
from os.path import isfile, join
def test_2():
    path = '../data/custom'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    n_right = 0
    n_tot = len(files)
    for file in files:
        test_img_value = filter(str.isdigit, file)
        n_right += load_and_predict_custom_image(filename,file, int(test_img_value))
    print(str(n_tot - n_right)+ ' wrong predictions out of ' + str(n_tot) )

import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Test 3
def test_3():
    wrongpredictions = predict_mlp_all_fast(filename, test_data='train', saveToFile=False, diagnose=False)
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

test_3()

def test_4(example_index):
    b = predict_mlp(filename, example_index, test_train_data = True)
    plt.title(str(example_index) + ' -> prediction = ' +str(b[0][0])+' , should be = '+str(b[0][1]),  fontsize=24)
    fig = plt.imshow(b[1].reshape((28, 28)).eval(), cmap = cm.Greys_r)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()









