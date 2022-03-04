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
    # Plotting the dependence of sub-solar temperature on heliocentric distance, according to Eq. (2) of Harris (1998)
    def calculate_subsolar_temperature(heliocentric_distance: float, albedo=0, emissivity=1, beaming_param=1):
        """
        Calculate subsolar temperature of an asteroid's surface, using Eq. (2) of article "A Thermal Model For Near
        Earth Asteroids", A. W. Harris (1998), the article that introduced NEATM.

        :param heliocentric_distance: float
            Distance from the Sun, in astronomical units
        :param albedo: float
            How much the asteroid reflects, between 0 and 1. For ideal blackbody this is 0.
        :param emissivity: float
            Emission from asteroid divided by emission from ideal blackbody.
        :param beaming_param: float
            Beaming parameter, the surface geometry / roughness effects compared to a perfect sphere.

        :return T_ss:
            Subsolar temperature, in Kelvin
        """
        T_ss = (((1 - albedo) * 1361 * (1 / heliocentric_distance**2)) / (beaming_param * emissivity * C.stefan_boltzmann))**0.25
        return T_ss

    d_S = np.linspace(0.5, 4)
    ss_temps_max = []
    ss_temps_min = []
    for distance in d_S:
        temperature_max = calculate_subsolar_temperature(distance)
        ss_temps_max.append(temperature_max)
        # temperature_min = calculate_subsolar_temperature(distance, albedo=0.9, emissivity=0.9, beaming_param=1)
        # ss_temps_min.append(temperature_min)
    plt.figure()
    plt.plot(d_S, ss_temps_max)
    # plt.plot(d_S, ss_temps_min)
    plt.xlabel('Heliocentric distance [AU]')
    plt.ylabel('Subsolar temperature [K]')
    plt.savefig(Path(C.figfolder, 'ss-temp_hc-dist.png'))
    plt.show()
    print('test')
    ############################
    # # Plotting an example radiance
    # num = 22
    # rads = FH.load_toml(Path(C.radiance_test_path, f'rads_{num}.toml'))
    # refrad = rads['reflected_radiance']
    # sumrad = rads['sum_radiance']
    # meta = rads['metadata']
    #
    # refR = rad.radiance2norm_reflectance(refrad)
    # sumR = rad.radiance2norm_reflectance(sumrad)
    #
    # plt.figure()
    # plt.plot(C.wavelengths, refrad)
    # plt.plot(C.wavelengths, sumrad)
    # plt.xlabel('Wavelength [µm]')
    # plt.ylabel('Radiance [W / m² / sr / µm]')
    # plt.legend(('Reference', 'With thermal'))
    #
    # plt.figure()
    # plt.plot(C.wavelengths, refR)
    # plt.plot(C.wavelengths, sumR)
    # plt.xlabel('Wavelength [µm]')
    # plt.ylabel('Normalized reflectance')
    # plt.legend(('Reference', 'With thermal'))
    #
    # plt.show()
    #
    # print('test')

    # # Build and train a model
    # model = NN.train_autoencoder(early_stop=True, checkpoints=True, save_history=True)

    # # Loading errors from Bennu testing, plotting results
    # errordict = FH.load_toml(Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/figs/validation_plots/validation-run_20220301-135222/bennu_validation/errors_Bennu.toml'))
    # errors_1000 = errordict['errors_1000']
    # errors_1230 = errordict['errors_1230']
    # errors_1500 = errordict['errors_1500']
    #
    # def Bennuplot(errors_1000, errors_1230, errors_1500, data_name, label, savefolder):
    #
    #     def fetch_data(errordict, data_name):
    #         temperature_dict = errordict['temperature']
    #         ground_temps = np.asarray(temperature_dict['ground_temperature'])
    #
    #         if 'MAE' in data_name:
    #             data_dict = errordict['MAE']
    #             if 'reflected' in data_name:
    #                 data = np.asarray(data_dict['reflected_MAE'])
    #             else:
    #                 data = np.asarray(data_dict['thermal_MAE'])
    #         elif 'SAM' in data_name:
    #             data_dict = errordict['SAM']
    #             if 'reflected' in data_name:
    #                 data = np.asarray(data_dict['reflected_SAM'])
    #             else:
    #                 data = np.asarray(data_dict['thermal_SAM'])
    #         elif 'temperature' in data_name:
    #             data_dict = errordict['temperature']
    #             if 'predicted' in data_name:
    #                 data = np.asarray(data_dict['predicted_temperature'])
    #             elif 'error' in data_name:
    #                 data = ground_temps - np.asarray(data_dict['predicted_temperature'])
    #
    #         return ground_temps, data
    #     ground_temps_1000, data_1000 = fetch_data(errors_1000, data_name)
    #     ground_temps_1230, data_1230 = fetch_data(errors_1230, data_name)
    #     ground_temps_1500, data_1500 = fetch_data(errors_1500, data_name)
    #
    #     plt.figure()
    #     plt.scatter(ground_temps_1000, data_1000, alpha=0.1)
    #     plt.scatter(ground_temps_1230, data_1230, alpha=0.1)
    #     plt.scatter(ground_temps_1500, data_1500, alpha=0.1)
    #     plt.xlabel('Ground truth temperature [K]')
    #     plt.ylabel(label)
    #     leg = plt.legend(('10:00', '12:30', '15:00'), title='Local time on Bennu')
    #     for lh in leg.legendHandles:
    #         lh.set_alpha(1)
    #     if data_name == 'predicted_temperature':
    #         plt.plot(range(300, 350), range(300, 350), 'r')  # Plot a reference line with slope 1: ideal result
    #     plt.savefig(Path(savefolder, f'{data_name}.png'))
    #     # plt.show()
    #
    # savefolder = C.validation_plots_path
    # Bennuplot(errors_1000, errors_1230, errors_1500, 'predicted_temperature', 'Predicted temperature [K]', savefolder)
    # Bennuplot(errors_1000, errors_1230, errors_1500, 'temperature_error', 'Temperature difference [K]', savefolder)
    # Bennuplot(errors_1000, errors_1230, errors_1500, 'reflected_MAE', 'Reflected MAE', savefolder)
    # Bennuplot(errors_1000, errors_1230, errors_1500, 'thermal_MAE', 'Thermal MAE', savefolder)
    # Bennuplot(errors_1000, errors_1230, errors_1500, 'reflected_SAM', 'Reflected SAM', savefolder)
    # Bennuplot(errors_1000, errors_1230, errors_1500, 'thermal_SAM', 'Thermal SAM', savefolder)
    # print('test')

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

    # last_epoch = 990
    # weight_path = Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5')
    weight_path = Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/training/300epochs_160waist_1e-05lr/weights/weights_297.hdf5')
    model.load_weights(weight_path)

    from contextlib import redirect_stdout

    timestr = time.strftime("%Y%m%d-%H%M%S")
    # timestr = 'test'  # folder name for test runs, otherwise a new folder is always created

    validation_run_folder = Path(C.validation_plots_path, f'validation-run_{timestr}')
    if os.path.isdir(validation_run_folder) == False:
        os.mkdir(validation_run_folder)

    # Print summary of model architecture into file
    with open(Path(validation_run_folder, 'modelsummary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    val.validate_synthetic(model, validation_run_folder)

    # Plotting some errors as function of ground truth temperature
    # folderpath = Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/figs/validation_plots/validation-run_20220224-110013/synthetic_validation')
    # val.error_plots(folderpath)

    ##############################

    # Testing with real asteroid data: do not look at this until the network works properly with synthetic data
    val.validate_bennu(model, validation_run_folder)






