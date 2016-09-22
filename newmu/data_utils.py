import cPickle as pickle

def save_model(params, epoch = 0, annotation='', namestub = '', savepath = '../data/models/', test_score=0.0):
    savedFileName = namestub + '_' + str(epoch)  + '_pars_' +  annotation +'.pkl'
    gg = open(savepath + savedFileName, 'wb')
    pickle.dump(params, gg, protocol=pickle.HIGHEST_PROTOCOL)
    gg.close()
    print(('Model params saved as ' + savedFileName
           + ' with test score %f %%') % (test_score * 100.))

def load_params(filename):
    gg = open(filename, 'rb')
    params = pickle.load(gg)
    gg.close()
    return params