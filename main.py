import random
from os import path
import os
import time
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from contextlib import redirect_stdout  # For saving keras prints into text files
# from astropy.io import fits
from scipy import io
import pandas as pd
from tensorflow import keras
import pickle
from sklearn.model_selection import train_test_split

import utils
import constants as C
import reflectance_data as refl
import radiance_data as rad
import file_handling as FH
import neural_network as NN
import validation as val  # TODO This uses symfit, which I have not installed on my thingfish conda env

# PyPlot settings to be used in all plots
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'savefig.dpi': 600})

if __name__ == '__main__':

    ############################
    # For running with GPU on server:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Check available GPU with command nvidia-smi in terminal, pick one that is not in use
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    # After you have started your computing task please use "nvidia-smi" command
    # and check that your program has correctly reserved GPU memory and that it
    # actually runs in GPU(s).
    #
    # Memory usage is in the middle column and GPU usage is in the rightmost co-
    # lumn. If GPU usage shows 0% then your code runs only in CPU, not in GPU.

    # To use plt.show() from server, make X11 connection and add this to Run configuration > Environment variables:
    # DISPLAY=localhost:10.0
    # But in most cases it would be better to just save the plots as png in a folder.
    ############################

    # Create a neural network model
    untrained = NN.create_model(
        conv_filters=60,
        conv_kernel=40,
        encdec_start=800,
        encdec_node_relation=0.5,
        waist_size=160,
        lr=1e-5
    )

    # # Load weights to continue training where you left off:
    # last_epoch = 295
    # weight_path = Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5')
    # untrained.load_weights(weight_path)

    # Train the model
    model = NN.train_autoencoder(untrained, early_stop=False, checkpoints=True, save_history=True, create_new_data=False)

    ##############################

    # Build a model and load pre-trained weights
    model = NN.create_model(
        conv_filters=60,
        conv_kernel=40,
        encdec_start=800,
        encdec_node_relation=0.5,
        waist_size=160,
        lr=1e-5
    )

    # last_epoch = 990
    # weight_path = Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5')
    weight_path = Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/training/300epochs_160waist_1e-05lr/weights/weights_297.hdf5')
    model.load_weights(weight_path)

    # Run validation with synthetic data and test with real data
    val.validate_and_test(model)

    ##############################

    # # Loading errors from Bennu testing, plotting results
    # errordict = FH.load_toml(Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/figs/validation_plots/validation-run_20220301-135222/bennu_validation/errors_Bennu.toml'))
    # val.plot_Bennu_errors(errordict)

    # Plotting some errors as function of ground truth temperature
    # folderpath = Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/figs/validation_plots/validation-run_20220224-110013/synthetic_validation')
    # val.error_plots(folderpath)

    ##############################








