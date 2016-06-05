**Some additional functionality to easily test prediction of custom images on [DeepLearningTutorials](http://deeplearning.net/tutorial/) DML examples**

The script *mlp_test/mlp_modified.py* is based on the [Multilayer perceptron](https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/mlp.py).
The script *mlp_test/convolutional_mlp_modified.py* is based on the [Deep Convolutional Network](https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/convolutional_mlp.py) - a simplified version of LeNet5.
Some functionality relying on [openmachinesblog](https://github.com/openmachinesblog/tensorflow-mnist/blob/master/mnist.py) code is used..
The scripts save the model parameters into a file named *best_model_mlp_(n-epochs).pkl* for *mlp_test* and *best_model_convolutional_mlp_(n-epochs).pkl* for *convolutional_mlp_modified*.
They also provide methods to load the saved parameters into the model and make predictions on custom images as well as onimages from the MNIST set.
The usage is illustrated in *mlp_test/test_mlp.py* and *mlp_test/test_lenet.py* respectively.

In order to test your own custom images, proceed as follows.
First run *mlp_modified* or *convolutional_mlp_modified* so as to obtain and save the relevant parameters.
Then drop the .png files containing your images into the *data/custom* directory.
Change the files titles so that they contain the target digit. Examples in data/custom are provided.
Run *mlp_test/test_*.py*.

Similarly for *SdA_modified.py* the the model parameters are saved into a file named *best_model_sda_(pretraining_epochs)_(training_epochs).pkl*

The *data/transform* directory contains the tranformed (MNISTized ...) files which are then processed by the model.

More at http://deepmachinelearning.blogspot.com