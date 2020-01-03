# Optretina's CNN architectures
Support material for Optretina's paper *Artificial intelligence to identify retinal fundus images, quality validation, laterality evaluation, macular degeneration, and suspected glaucoma*.

This repository includes an implementation of the CNN architectures presented in the paper. The default parameters correspond to those we used in the paper.

We've also included code to evaluate trained Keras models and calculate the published metrics.


## Environment setup
You can use conda to install all related dependencies to run this project.

To create the environment run:

    $ conda env create -n optretina-cnns -f environment.yml


## Architecture model usage
`models.py` contains the three architectures described in the paper implemented in Keras. It is possible to create a Keras model just by calling the function name, along with the number of output classes. For example:

	In [1]: import models
	In [2]: cnn1_model = models.CNN1(4)  # You can specify the number of output nodes
	In [3]: cnn1_model.load_weights("weights/type_model.h5")
	In [4]: amdnet_model = models.AMDNet(2, (512, 512, 3))  # You can also specify image input size
	In [5]: amdnet_model.load_weights("weights/dmae_model_20181204.h5")
	In [6]: gon_model = models.GONNet(2)
	In [7]: gon_model.load_weights("weights/glaucoma_model_20181230.h5")

The optimizer has been set by default to use the parameters employed during training. The default input size of the images is also that which is presented in the paper.


## Keras model evaluation
`testing.py` can be used to evaluate a trained Keras model on a set of images. The output is a CSV file that will contain the score per class. This CSV output can later be used to compute the reported metrics in the paper.

	$ python testing.py 
	usage: testing.py [-h] [--color {rgb,grayscale}] [--batch-size BATCH_SIZE]
	                  [--column COLUMN]
	                  images_path crop_size model_path csv_output
	testing.py: error: the following arguments are required: images_path, crop_size, model_path, csv_output


## Calculate reported metrics and figures
