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
from contextlib import redirect_stdout  # For saving keras prints into text files
# from astropy.io import fits
from scipy import io
import pandas as pd
from tensorflow import keras
import pickle
from sklearn.model_selection import train_test_split
import keras_tuner as kt

from solar import solar_irradiance
import constants as C
import reflectance_data as refl
import radiance_data as rad
import file_handling as FH
import neural_network as NN
import validation as val  # TODO This uses symfit, which I have not installed on my thingfish conda env


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
    """
    Three best models from hp-optimizer:
    
        Trial summary
        Hyperparameters:
        filters: 60
        kernel_size: 40
        encdec_start: 800
        waist_size: 160
        encdec_node_relation: 0.1
        lr: 1e-05
        Score: 0.19715037941932678
        
        Trial summary
        Hyperparameters:
        filters: 20
        kernel_size: 40
        encdec_start: 1200
        waist_size: 160
        encdec_node_relation: 0.5
        lr: 1e-05
        Score: 0.19769485294818878
        
        Trial summary
        Hyperparameters:
        filters: 60
        kernel_size: 40
        encdec_start: 1200
        waist_size: 160
        encdec_node_relation: 0.1
        lr: 1e-05
        Score: 0.19800615310668945
    """

    # untrained = NN.create_model(
    #     conv_filters=60,
    #     conv_kernel=40,
    #     encdec_start=800,
    #     encdec_node_relation=0.5,
    #     waist_size=160,
    #     lr=1e-5
    # )
    #
    # # Load weights to continue training where you left off:
    # last_epoch = 295
    # weight_path = Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5')
    # untrained.load_weights(weight_path)
    #
    # model = NN.train_autoencoder(untrained, early_stop=False, checkpoints=True, save_history=True, create_new_data=False)

    ##############################

    # # Hyperparameter optimization with KerasTuner
    # # hypermodel = NN.create_hypermodel(kt.HyperParameters())
    # savefolder_name = f'optimization-run_{time.strftime("%Y%m%d-%H%M%S")}'
    # tuner = kt.BayesianOptimization(
    #     hypermodel=NN.create_hypermodel,
    #     objective="val_loss",
    #     max_trials=20,
    #     executions_per_trial=1,
    #     overwrite=True,
    #     directory=C.hyperparameter_path,
    #     project_name=savefolder_name,
    # )
    # tuner.search_space_summary()
    # x_train, y_train, x_val, y_val = NN.load_training_validation_data()
    # tuner.search(x_train, y_train, epochs=300, validation_data=(x_val, y_val))
    #
    # tuner.results_summary()
    # tuning_results_path = Path(C.hyperparameter_path, savefolder_name)
    # with open(Path(tuning_results_path, 'trial_summary.txt'), 'w') as f:
    #     with redirect_stdout(f):
    #         tuner.results_summary()

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

    last_epoch = 297
    weight_path = Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5')
    model.load_weights(weight_path)
    #
    #
    from contextlib import redirect_stdout
    #
    timestr = time.strftime("%Y%m%d-%H%M%S")
    validation_run_folder = Path(C.validation_plots_path, f'validation-run_{timestr}')
    if os.path.isdir(validation_run_folder) == False:
        os.mkdir(validation_run_folder)

    # Print summary of model architecture into file
    with open(Path(validation_run_folder, 'modelsummary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    # #
    # val.validate_synthetic(model, last_epoch, validation_run_folder)

    # Plotting some errors as function of ground truth temperature
    # folderpath = Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/figs/validation_plots/validation-run_20220224-110013/synthetic_validation')
    # val.error_plots(folderpath)

    ##############################

    # Validation with real asteroid data: do not look at this until the network works properly with synthetic data
    val.validate_bennu(model, last_epoch, validation_run_folder)






