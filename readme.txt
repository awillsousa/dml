The script mlp_test/mlp_modified is based on https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/mlp.py
Some functionality relying on https://github.com/openmachinesblog/tensorflow-mnist/blob/master/mnist.py has been added.
The script saves the model parameters into a file best_model_(n-epochs).pkl.
It also providss methods to load the saved parameters into the model and make predictions on images from the MNIST set
and on custom images as well.
The usage is illustrated in mlp_test/test.py.

In order to test your own custom images, proceed as follows.
First run mlp_modified so as to obtain and save the relevant parameters.
Then drop the .png files containing your images into data/custom.
Change their title so that they contain the target digt. Examples in data/custom are provided.
Run mlp_test/test.py.

The data/tranform directory contains the tranformed (MNISTized ...) files which are then processed by the model.

