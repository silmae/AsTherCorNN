
import os

import numpy as np
from matplotlib import pyplot as plt

# PyPlot settings to be used in all plots
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'savefig.dpi': 600})
# LaTeX font for all text in all figures
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

import utils
import constants as C
import reflectance_data as refl
import radiance_data as rad
import file_handling as FH
import neural_network as NN
import validation as val

if __name__ == '__main__':

    ############################
    # For running with GPU on server (having these lines here shouldn't hurt when running locally without GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Check available GPU with command nvidia-smi in terminal, pick one that is not in use
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    ############################
    # HIEKKALAATIKKO
    maxtemp = utils.calculate_subsolar_temperature(C.d_S_min + 0.1)
    maxtemp = utils.calculate_subsolar_temperature(C.d_S_min)
    val.error_plots('./validation_and_testing/validation-run_epoch-470_time-test/synthetic_validation')
    val.plot_Bennu_errors('./validation_and_testing/validation-run_epoch-470_time-test/bennu_validation')

    ############################
    # # HYPERPARAMETER OPTIMIZATION
    # NN.tune_model(300, 20, 1)

    ############################
    # # TRAINING
    # # NN.prepare_training_data()
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

    ##############################
    # # VALIDATION
    # # Run validation with synthetic data and test with real data
    # last_epoch = 470
    # val.validate_and_test(last_epoch)
