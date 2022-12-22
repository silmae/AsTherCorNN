
import os
import utils
import csv
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

import constants as C
import reflectance_data as refl
import radiance_data as rad
import file_handling as FH
import neural_network as NN
import validation as val

# PyPlot settings to be used in all plots
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'savefig.dpi': 600})
# LaTeX font for all text in all figures
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
# Set marker size for scatter plots
plt.rcParams['lines.markersize'] = 4


if __name__ == '__main__':

    ############################
    # For running with GPU on server (having these lines here shouldn't hurt when running locally without GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Check available GPU with command nvidia-smi in terminal, pick one that is not in use
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    ############################
    # PRELIMINARIES

    # # Plot percentage error caused by thermal emission at 1 au as a function of temperature, for three albedo values
    # utils.thermal_error_from_temperature(albedo_min=0.02,
    #                                      albedo_max=0.08,
    #                                      temperature_min=250,
    #                                      temperature_max=350,
    #                                      hc_distance=1,
    #                                      samples=100,
    #                                      log_y=False)
    #
    # # Plotting approximate reflectance error at 2.45 µm caused by thermal emission as function of heliocentric distance
    # utils.thermal_error_from_hc_distance(distance_min=1, distance_max=3, samples=20, log_y=False)
    #
    # # Maximum temperature: subsolar temperature of ideal blackbody placed at the perihelion distance given in
    # # constants.py (d_S_min)
    # maxtemp = utils.calculate_subsolar_temperature(C.d_S_min)

    # # Create training and validation data
    # NN.prepare_training_data()

    ############################
    # # HYPERPARAMETER OPTIMIZATION
    # NN.tune_model(300, 20, 1)

    ############################
    # # TRAINING
    # # Create a neural network model
    # untrained = NN.create_model(
    #     conv_filters=C.conv_filters,
    #     conv_kernel=C.conv_kernel,
    #     encoder_start=C.encoder_start,
    #     encoder_node_relation=C.encoder_node_relation,
    #     encoder_stop=C.encoder_stop,
    #     lr=C.learning_rate
    # )
    #
    # # # Load weights to continue training where you left off:
    # # last_epoch = 445
    # # weight_path = Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5')
    # # untrained.load_weights(weight_path)
    #
    # # Train the model
    # model = NN.train_network(untrained, early_stop=False, checkpoints=True, save_history=True, create_new_data=False)

    # # Plot training and validation loss history saved in a log file
    # NN.plot_loss_history(3)

    ##############################
    # VALIDATION / TEST
    # # Run validation with synthetic data and test with real data
    # last_epoch = 29
    # val.validate_and_test(last_epoch)

    # Make new plots from existing validation/test results
    val.error_plots('./validation_and_testing/validation-run_epoch-29_time-20221219-121903/synthetic_validation')
    val.plot_Bennu_errors('./validation_and_testing/validation-run_epoch-29_time-20221219-121903/bennu_validation')
