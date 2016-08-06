import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import gzip

filename = "../data/mnist.pkl.gz"
f = gzip.open(filename, 'rb')
train_data = pickle.load(f)[0]
f.close()