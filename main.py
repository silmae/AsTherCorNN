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
    # # TRAINING
    # # Create a neural network model
    # untrained = NN.create_model(
    #     conv_filters=60,
    #     conv_kernel=40,
    #     encdec_start=800,
    #     encdec_node_relation=0.5,
    #     waist_size=160,
    #     lr=1e-5
    # )
    #
    # # # Load weights to continue training where you left off:
    # # last_epoch = 295
    # # weight_path = Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5')
    # # untrained.load_weights(weight_path)
    #
    # # Train the model
    # model = NN.train_autoencoder(untrained, early_stop=False, checkpoints=True, save_history=True, create_new_data=False)

    ##############################

    # # VALIDATION
    # # Build a model and load pre-trained weights
    # model = NN.create_model(
    #     conv_filters=60,
    #     conv_kernel=40,
    #     encdec_start=800,
    #     encdec_node_relation=0.5,
    #     waist_size=160,
    #     lr=1e-5
    # )
    #
    # last_epoch = 1
    # weight_path = Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5')
    # # weight_path = Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/training/300epochs_160waist_1e-05lr/weights/weights_297.hdf5')
    # model.load_weights(weight_path)
    #
    # # Run validation with synthetic data and test with real data
    # val.validate_and_test(model)

    ##############################

    # # Loading errors from Bennu testing, plotting results
    # errordict = FH.load_toml(Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/figs/validation_plots/validation-run_20220301-135222/bennu_validation/errors_Bennu.toml'))
    # val.plot_Bennu_errors(errordict)

    # Plotting some errors as function of ground truth temperature
    # folderpath = Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/figs/validation_plots/validation-run_20220224-110013/synthetic_validation')
    # val.error_plots(folderpath)

    ##############################

    # FINDING BOUNDARY CONDITIONS FOR HELIOCENTRIC DISTANCE
    # Looking at a worst-case scenario: dark asteroid (T class, p = 0.03), temperature according to subsolar temp of
    # an ideal blackbody.

    # Loading a reflectance spectrum of type T asteroid
    aug_path = C.Penttila_aug_path  # Spectra augmented by Penttilä

    aug_frame = pd.read_csv(aug_path, sep='\t', header=None, engine='python')  # Read wl and reflectance from file
    albedo_frame = pd.read_csv(C.albedo_path, sep='\t', header=None, engine='python', index_col=0)  # Read mean albedos for classes

    for row in aug_frame.values:
        # The first value of a row is the asteroid class, the rest is normalized reflectance
        asteroid_class, norm_reflectance = row[0], row[1:]

        # Take the first encountered spectrum of class T and scale it with the class minimum albedo
        if asteroid_class == 'T':
            # Fetch the asteroid class albedo and its range. Take three values using the min, mid, and max of the range
            alb = albedo_frame.loc[asteroid_class].values
            geom_albedo = alb[0] - 0.5*alb[1]

            # Convert geometrical albedo to Bond albedo, assuming Lommel-Seeliger TODO Formula by Penttilä, find a reference or make it yoself
            bond_albedo = 16 * geom_albedo * (1 - np.log(2)) / 3

            # Scale by multiplying with p/mean(norm_refl): values stay between 0 and 1, mean of scaled vector will be p
            reflectance = norm_reflectance * (bond_albedo / np.mean(norm_reflectance))

            # Print if the physical limits of min and max reflectance are exceeded
            if np.max(reflectance) > 1 or np.min(reflectance) < 0:
                print(f'Unphysical reflectance detected! Max {np.max(reflectance)}, min {np.min(reflectance)}')
            if np.mean(reflectance) - bond_albedo > 0.001:
                print(f'Deviation from albedo detected! Difference between albedo and mean R {np.mean(reflectance) - bond_albedo}')
            break

    # Calculate spectral emittance with Kirchhoff's law
    emittance = 1 - reflectance

    # A list of heliocentric distances
    distances = np.linspace(2.0, 5.0, 50)
    # A list of temperatures
    temperatures = utils.maximum_temperatures(2.0, 5.0, 50)
    # Empty list for storing errors
    errors = []

    i = 0
    for distance in distances:
        insolation = utils.solar_irradiance(distance, C.wavelengths)
        radiance_dict = rad.observed_radiance(d_S=distance,
                                              incidence_ang=0,
                                              emission_ang=0,
                                              T=temperatures[i],
                                              reflectance=reflectance,
                                              waves=C.wavelengths,
                                              filename='filename',
                                              test=True,
                                              save_file=False)
        reflected_radiance = radiance_dict['reflected_radiance']
        # thermal_radiance = radiance_dict['emitted_radiance']
        sum_radiance = radiance_dict['sum_radiance']
        i = i + 1

        # Calculate normalized reflectance from sum radiance and reflected radiance
        reference_reflectance = rad.radiance2norm_reflectance(reflected_radiance)
        test_reflectance = rad.radiance2norm_reflectance(sum_radiance)

        # Take the last element of both vectors and calculate the difference as percentage
        reference = reference_reflectance[-1]
        test = test_reflectance[-1]
        error = 100 * (abs(reference - test)) / reference
        errors.append(error)

        # Plotting radiances and reflectances
        fig = plt.figure()
        plt.plot(C.wavelengths, reflected_radiance)
        plt.plot(C.wavelengths, sum_radiance)
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Radiance [W / m² / sr / µm]')
        plt.legend(('Reference', 'Test'))
        plt.savefig(Path(C.max_temp_plots_path, f'{i}_radiance.png'))
        plt.close(fig)

        fig = plt.figure()
        plt.plot(C.wavelengths, reference_reflectance)
        plt.plot(C.wavelengths, test_reflectance)
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Normalized reflectance')
        plt.legend(('Reference', 'Test'))
        plt.savefig(Path(C.max_temp_plots_path, f'{i}_reflectance.png'))
        plt.close(fig)


    # Plot error as function of heliocentric distance
    plt.figure()
    plt.plot(distances, errors)
    plt.xlabel('Heliocentric distance [AU]')
    plt.ylabel('Reflectance error at 2.5 µm [%]')
    plt.savefig(Path(C.max_temp_plots_path, f'{i}_error_hc-distance.png'))
    plt.show()







