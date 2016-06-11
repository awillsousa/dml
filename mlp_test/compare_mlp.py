import cPickle as pickle
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


path = '../data/models/'

def get_filenames(startString, endString,  path = '../data/models' ):
    files = [f for f in listdir(path) if isfile(join(path, f)) and startString in f and f.endswith(endString)]
    theKey = lambda f: int(filter(str.isdigit, f))
    thefiles = sorted([f for f in files], key=theKey)
    return thefiles


#model_and_filename resp. models_and_filenames have the structure [[W,b,W,b], filename]
def calculate_distances_from_model(model_and_filename, models_and_filenames):
    results = []
    n_models = len(models_and_filenames);
    print ('Processing ' + str(n_models) + ' models.')
    for p in range(n_models):
        pdigit = int(filter(str.isdigit, models_and_filenames[p][1]))
        result = [pdigit,]
        for j in range(2):
            dW = norm(models_and_filenames[p][0][2 * j] - model_and_filename[0][2 * j])
            dB = norm(models_and_filenames[p][0][2 * j + 1] - model_and_filename[0][2 * j + 1])
            result.append(dW.eval())
            result.append(dB.eval())
        results.append(result)
    return [model_and_filename[1], results]

def calculate_distance_pairs(models_and_filenames_1, models_and_filenames_2):
    results = []
    n_models_1 = len(models_and_filenames_1)
    n_models_2 = len(models_and_filenames_2)
    if n_models_1 >= n_models_2:
        n_models = n_models_2
    else:
        n_models = n_models_1
    print ('Processing ' + str(n_models) + ' models.')
    for p in range(n_models):
        print(str(p) +' - ' + models_and_filenames_1[p][1])
        pdigit = int(filter(str.isdigit, models_and_filenames_1[p][1]))
        result = [pdigit, ]
        for j in range(2):
            dW = norm(models_and_filenames_1[p][0][2 * j] - models_and_filenames_2[p][0][2 * j])
            dB = norm(models_and_filenames_1[p][0][2 * j + 1] - models_and_filenames_2[p][0][2 * j + 1])
            result.append(dW.eval())
            result.append(dB.eval())
        results.append(result)
    return results

def load_model(fname, path = '../data/models/'):
    gg = open(path + fname, 'rb')
    model = pickle.load(gg)
    gg.close()
    return model, fname


def load_models(filenames):
    models = []
    for fname in filenames:
        models.append(load_model(fname))
    return models

def get_value_from_distances(dist, index):
    return [dist[i][index] for i in range(0, len(dist))]

def norm(x):
    return ((x ** 2).sum()) ** 0.5

def plot_distance_pairs(distances):
    epoch = get_value_from_distances(distances, 0)

    W1 = get_value_from_distances(distances, 1)

    b1 = get_value_from_distances(distances, 2)

    W2 = get_value_from_distances(distances, 3)

    b2 = get_value_from_distances(distances, 4)

    fig, ax = plt.subplots()

    ax.plot(epoch, W1, 'c--', label='HiddenLayer W')
    ax.plot(epoch, b1, 'b--', label='HiddenLayer b')
    ax.plot(epoch, W2, 'g--', label='RegressionLayer W')
    ax.plot(epoch, b2, 'r--', label='RegressionLayer b')

    plt.legend = ax.legend(loc='upper right', shadow=True)

    plt.xlabel("epoch")
    plt.ylabel("distance between zero models and rand models")

    plt.show()


def plot_distances_from_target(themodel, themodels):
    distances_and_name = calculate_distances_from_model(themodel, themodels)
    target_name = distances_and_name[0]
    distances = distances_and_name[1]

    epoch = get_value_from_distances(distances, 0)

    W1 = get_value_from_distances(distances, 1)

    b1 = get_value_from_distances(distances, 2)

    W2 = get_value_from_distances(distances, 3)

    b2 = get_value_from_distances(distances, 4)

    fig, ax = plt.subplots()

    ax.plot(epoch, W1, 'c--', label='HiddenLayer W')
    ax.plot(epoch, b1, 'b--', label='HiddenLayer b')
    ax.plot(epoch, W2, 'g--', label='RegressionLayer W')
    ax.plot(epoch, b2, 'r--', label='RegressionLayer b')

    plt.legend = ax.legend(loc='upper right', shadow=True)

    plt.xlabel("epoch")
    plt.ylabel("distance from " + target_name)

    plt.show()


#plot_distances_from_target('best_model_mlp_500.pkl', get_filenames('best_model_mlp')[0] )

def calculate_distance(filenames):
    models = load_models(filenames)
    results = []
    for p1 in range(len(models)):
        p1digit = int(filter(str.isdigit, filenames[p1]))
        for p2 in range(p1 + 1, len(models)):
            p2digit = int(filter(str.isdigit, filenames[p2]))
            for j in range(2):
                dW = norm(models[p1][2 * j] - models[p2][2 * j])
                dB = norm(models[p1][2 * j + 1] - models[p2][2 * j + 1])
                results.append([p1digit, p2digit, j, dW.eval(), dB.eval()])
    return results



def test_descending_distances(results):
    for p1 in range(len(results)):
        for p2 in range(p1 + 1, len(results)):
            if (results[p1][0] <= results[p2][0] and results[p1][1] >= results[p2][1]
                and results[p1][2] == results[p2][2]):
                exp = (results[p1][3] > results[p2][3] and results[p1][4] > results[p2][4])
                if not exp: raise AssertionError(' p1 = ' + str(results[p1]) + ' p1 = ' + str(results[p2]))
                print(str(p1) + ':' + str(p2) + ' - OK')


import os
files = [f for f in listdir('.') if isfile(join('.', f)) and 'best_model_convolutional_mlp' in f and not '_test' in f]
for file in files:
    print('a - ' + file)
    file_n = filter(str.isdigit, file)
    print('b - ' +str(file_n))
    index = file.find(file_n)
    print('c - ' +file[:index])
    print('d - ' + file[index + len(file_n):])
    print('e - '+ file[:index]+ file_n +'_zero'+ file[index + len(file_n):])
    newname = file[:index]+ file_n +'_zero'+ file[index + len(file_n):]
    os.rename(file, newname)

