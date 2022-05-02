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

# PyPlot settings to be used in all plots
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'savefig.dpi': 600})
# plt.rcParams['text.usetex'] = True

if __name__ == '__main__':

    # # Testing some random stuff that has nothing to do with this
    # def channelCM_from_RGB(dirpath: str, filename: str):
    #     """
    #     Loads an RGB image (.png or .jpg, other formats might work also) and creates colormaps for the R, G, and B
    #     channels. Saves the colormaps as .png images in the same folder where the RGB image was loaded from.
    #     :param dirpath:
    #         Path to the directory where the image is located. As a STRING, not as a path!
    #     :param filename:
    #         Filename of the image as a string, with extension
    #     """
    #     import matplotlib.image as img
    #     filepath = dirpath + filename
    #     file = img.imread(filepath)
    #
    #     R = file[:, :, 0]
    #     G = file[:, :, 1]
    #     B = file[:, :, 2]
    #
    #     def plot_cm(data, savepath):
    #         plt.figure()
    #         plt.imshow(data)
    #         plt.axis('off')
    #         plt.savefig(savepath, bbox_inches='tight')
    #
    #     plot_cm(R, f"{filepath[:-4]}R.png")
    #     plot_cm(G, f"{filepath[:-4]}G.png")
    #     plot_cm(B, f"{filepath[:-4]}B.png")
    #
    #     plt.show()
    #
    # channelCM_from_RGB('/home/leevi/PycharmProjects/pythonProject/figs/', 'muki.jpg')

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
    # HYPERPARAMETER OPTIMIZATION
    # NN.tune_model(300, 20, 1)

    ############################
    # # # TRAINING
    # # NN.prepare_training_data()
    # # Create a neural network model
    # untrained = NN.create_model(
    #     conv_filters=C.conv_filters,
    #     conv_kernel=C.conv_kernel,
    #     encdec_start=C.encdec_start,
    #     encdec_node_relation=C.encdec_node_relation,
    #     waist_size=C.waist,
    #     lr=C.learning_rate
    # )
    #
    # # Load weights to continue training where you left off:
    # last_epoch = 520
    # weight_path = Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5')
    # untrained.load_weights(weight_path)
    #
    # # Train the model
    # model = NN.train_autoencoder(untrained, early_stop=False, checkpoints=True, save_history=True, create_new_data=False)

    ##############################
    # VALIDATION
    import validation as val  # TODO This uses symfit, which I have not installed on my thingfish conda env

    # Build a model and load pre-trained weights
    model = NN.create_model(
        conv_filters=C.conv_filters,
        conv_kernel=C.conv_kernel,
        encdec_start=C.encdec_start,
        encdec_node_relation=C.encdec_node_relation,
        waist_size=C.waist,
        lr=C.learning_rate
    )

    last_epoch = 216
    weight_path = Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5')
    # weight_path = Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/training/300epochs_160waist_1e-05lr/weights/weights_297.hdf5')
    model.load_weights(weight_path)
    #
    # Run validation with synthetic data and test with real data
    val.validate_and_test(model)

    # val.error_plots(Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/validation_and_testing/validation-run_20220330-162708/synthetic_validation'))
    # val.plot_Bennu_errors('//home/leevi/PycharmProjects/asteroid-thermal-modeling/validation_and_testing/validation-run_20220421-103518/bennu_validation')
    #############################
    #
    # # Loading errors from Bennu testing, plotting results
    # errordict = FH.load_toml(Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/figs/validation_plots/validation-run_20220301-135222/bennu_validation/errors_Bennu.toml'))
    # val.plot_Bennu_errors(errordict)
    #
    # Plotting some errors as function of ground truth temperature
    # folderpath = Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/figs/validation_plots/validation-run_20220224-110013/synthetic_validation')
    # val.error_plots(folderpath)

    ##############################








