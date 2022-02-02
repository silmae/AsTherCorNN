import random
from os import path
import os
import time
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'savefig.dpi': 600})
# from astropy.io import fits
from scipy import io
import pandas as pd
from tensorflow import keras
import pickle
from sklearn.model_selection import train_test_split

from solar import solar_irradiance
import constants as C
import reflectance_data as refl
import radiance_data as rad
import toml_handler as tomler
import neural_network as NN
import validation as val


if __name__ == '__main__':

    #######################
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
    ############################

    # # Build and train a model
    # model = NN.train_autoencoder(early_stop=True, checkpoints=True, save_history=True)
    model = NN.train_autoencoder(early_stop=False, checkpoints=True, save_history=True, create_new_data=False)

    ##############################

    # # Build a model and load pre-trained weights
    # last_epoch = 9866
    # model = NN.load_model(Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5'))
    #
    # from contextlib import redirect_stdout
    #
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # validation_run_folder = Path(C.validation_plots_path, f'validation-run_{timestr}')
    # os.mkdir(validation_run_folder)
    #
    # # Print summary of model architecture into file
    # with open(Path(validation_run_folder, 'modelsummary.txt'), 'w') as f:
    #     with redirect_stdout(f):
    #         model.summary()
    # #
    # val.validate_synthetic(model, last_epoch, validation_run_folder)

    ##############################

    # # Validation with real asteroid data: do not look at this until the network works properly with synthetic data
    # val.validate_bennu(model, last_epoch, validation_run_folder)






