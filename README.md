# AsTherCorNN - Asteroid Thermal Correction Neural Network

This repository contains Python code for simulating spectral radiances from
asteroid surfaces, and for training a convolutional neural network to
predict surface temperatures from said radiances. The temperature predictions
are used to correct spectral radiance observations for thermal contribution,
in order to accurately determine the spectral reflectance. 
The software is published under the MIT license. 
This repository is related to a paper currently in the process of writing. 


## Working principle

The workings of the code could be roughly divided into three parts:
data generation, training a network, and testing the trained network. 
In broader terms the idea is to feed a spectral radiance observed from 
an asteroid's surface to a neural network and receive a temperature prediction
for the observed area as output. This can then be used together with an
emissivity value to get a prediction for thermally emitted spectral radiance.
Subtracting the thermal spectrum from the original sum spectrum yields
an approximation for reflected radiance. 

### Data generation
The training data consists of spectral radiances in the wavelengths 
0.45 - 2.45 µm, and ground truth temperature values corresponding to the 
spectra radiances. The radiances are generated with a very rudimentary 
simulator, which models total spectral radiance as a sum of reflected
and thermally emitted radiances. Thermal radiance is produced with Planck's 
law, plugging in a temperature value and multiplying the final result by 
an emissivity. Reflected radiances are modeled with the Lommel-Seeliger law,
which requires incident spectral irradiance, spectral single-scattering albedo, 
and incidence and emission angles. Single-scattering albedos are produced from
normalized reflectance spectra of asteroids produced in this study by Penttilä 
et al.: https://doi.org/10.1051/0004-6361/202038545. 
For training purposes the sum spectral radiances are the training samples, and the
temperatures used to produce the thermal radiances are the ground truth.

### Neural network
The neural network input is a spectral radiance with 200 channels from
wavelength range 0.45 - 2.45 µm. The output is a temperature value in kelvin.
These determine the input layer to be 200 units wide, and the output 1 unit.
The architecture of the hidden layers is based on a convolutional neural network (CNN), with 
a number of 1D convolution layers followed by dense layers.
The architecture was optimized with KerasTuner, and methods for this are
also included.

### Testing
Testing the network was done with synthetic data also used for validation
during training, and with real OVIRS observations of Bennu. The latter
were produced in this article: https://doi.org/10.1126/science.abc3522,
and were graciously provided to us by the author, Dr. Amy A. Simon.
Temperature predictions by the network were compared to ground truth data,
and radiances and reflectances corrected using the predictions to 
ones corrected with the ground truth values. Results of a test run
are presented mostly through various plots.

## Dependencies
The code is written entirely in Python, using Conda for managing packages. 
The tool used for creating the neural network was TensorFlow with its Keras 
framework.
The various packages and modules and their 
versions used in this project are listed in the file
`environment.yml`, which can be used to generate a 
matching Conda environment. 

N.B. For running the neural network code
on a GPU, you will need a different version of Tensorflow than the one listed 
in `environment.yml`. The listed version only supports CPU execution.


## Contents
The code is divided into several files, the contents of which are broadly 
as follows: 
- `constants.py` - all constants (or at least most of them), including physical constants, parameters for data generation and neural network architecture, and paths. The paths lay out a folder structure which users may re-create on their own machine.
- `file_handling.py` - methods for reading files from disc or writing into files
- `main.py` - main method for running the program, includes some example calls to the various methods
- `neural_network.py` - building a neural network, training it, and optimizing the architecture
- `radiance_data.py` - producing a radiance dataset for training, by simulating reflected and thermally emitted spectral radiances
- `reflectance_data.py` - methods for working with asteroid reflectance data
- `utils.py` - utility methods which did not fit into other modules
- `validation.py` - testing the performance of a trained neural network

The `training`-folder has in its subfolder the final weights of the trained 
neural network, stored in a format compatible with Keras. The folder also 
contains a log of the loss history for the final training run. 


## Citing this work
This work has currently not been published in a peer reviewed journal, but 
hopefully will be shortly. If you find the code published here useful in your
own research and wish to cite it, please check back here soon for 
info on how to do that!