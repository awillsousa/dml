import cPickle as pickle
import convolutional_mlp_modified as cmlp


paramsFilename = 'best_model_convolutional_mlp_100.pkl'


cmlp.predict_all_mnist_test_images(paramsFilename)