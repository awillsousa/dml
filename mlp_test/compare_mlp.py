import cPickle as pickle
from numpy import linalg as LA

# filenames0 = 'best_model_mlp_10.pkl','best_model_mlp_25.pkl','best_model_mlp_100.pkl', 'best_model_mlp_500.pkl'
filenames0 = 'best_model_mlp_10_rand.pkl', 'best_model_mlp_25_rand.pkl', 'best_model_mlp_100_rand.pkl'

models0 = []
for fname in filenames0:
    gg = open(fname, 'rb')
    models0.append(pickle.load(gg))
    gg.close()


def norm(x):
    return ((x ** 2).sum()) ** 0.5


def calculate_distance(models, filenames):
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
                    exp = (results[p1][3]>results[p2][3] and results[p1][4]>results[p2][4])
                    if not exp: raise AssertionError(' p1 = ' + str(results[p1]) + ' p1 = ' + str(results[p2]))
                    print(str(p1) + ':' + str(p2) + ' - OK')

#test_descending_distances(calculate_distance(models0, filenames0));

#print(str(calculate_distance(models0, filenames0)))