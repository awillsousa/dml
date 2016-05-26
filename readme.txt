The script mlp_test/mlp_modified.py is based on https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/mlp.py
The script mlp_test/convolutional_mlp_modified.py is based on https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/convolutional_mlp.py
Some functionality relying on https://github.com/openmachinesblog/tensorflow-mnist/blob/master/mnist.py has been added.
The scripts save the model parameters into a file named 'best_model_mlp_(n-epochs).pkl' for mlp_test and best_model_convolutional_mlp_(n-epochs).pkl for convolutional_mlp_modified.
They also provide methods to load the saved parameters into the model and make predictions on on custom images as well as onimages from the MNIST set.
The usage is illustrated in mlp_test/test_mlp.py and mlp_test/test_lenet.py respectively.

In order to test your own custom images, proceed as follows.
First run mlp_modified so as to obtain and save the relevant parameters.
Then drop the .png files containing your images into data/custom.
Change their title so that they contain the target digt. Examples in data/custom are provided.
Run mlp_test/test_*.py.

The data/tranform directory contains the tranformed (MNISTized ...) files which are then processed by the model.

