"""
This file houses all the constants, including the physical constants and paths, and such.
"""

import os
from pathlib import Path
import numpy as np
import scipy.constants

##########################################################################
# Physical constants, fetched from scipy library
c = scipy.constants.c  # 2.998e8 m / s, speed of light in vacuum
kB = scipy.constants.Boltzmann  # 1.381e-23 m² kg / s² / K (= J / K), Boltzmann constant
h = scipy.constants.h  # 6.626e-34 m² kg / s (= J s), Planck constant
stefan_boltzmann = scipy.constants.Stefan_Boltzmann  # 5.67e-8 W / m² / K⁴, Stefan-Boltzmann constant

##########################################################################
# Wavelength vector, the same as in used asteroid reflectances
step = 0.01  # µm
wavelengths = np.arange(0.46, 2.45 + step, step=step)

##########################################################################
# Paths
spectral_path = Path('./spectral_data')
Penttila_orig_path = Path(
    './spectral_data/reflectances/Penttila_asteroid_spectra/MyVISNIR-final-sampled-collection.dat')
'''Reflectance spectra of asteroids'''
Penttila_aug_path = Path(
    './spectral_data/reflectances/Penttila_asteroid_spectra/MyVISNIR-simulated-simplified-taxonomy.dat')
'''Reflectance spectra of asteroids, augmented'''
albedo_path = Path('./spectral_data/reflectances/Penttila_asteroid_spectra/class-mean-albedos.tab')
'''Mean albedos of asteroid spectral classes'''
solar_path = Path('./spectral_data/solar-spectral-irradiance/solar_spectrum.txt')
'''Solar irradiance spectrum'''
rad_bunch_test_path = Path('./spectral_data/rad_bunch_test')
'''All synthetic test radiances, saved as a pickle'''
rad_bunch_training_path = Path('./spectral_data/rad_bunch_training')
'''All training data, saved as a pickle'''
radiance_path = Path('./spectral_data/radiances')
'''All modeled radiances as toml files, separated in subdirectories for training and test'''
radiance_training_path = Path(radiance_path, 'training')
'''Training radiances saved as toml files with metadata'''
radiance_test_path = Path(radiance_path, 'test')
'''Test radiances saved as toml files with metadata'''

figfolder = Path('./figs')
'''Folder where most figures are saved (not the ones produced during model validation)'''
refl_plots_path = Path(figfolder, 'asteroid-reflectance-plots')
'''Plots of asteroid reflectances after un-normalization'''
rad_plots_path = Path(figfolder, 'radiance-plots')
'''Plots of modeled spectral radiances'''
max_temp_plots_path = Path(figfolder, 'max_temp_plots')
'''Plots related to maximum surface temperature evaluation'''
bennu_plots_path = Path(figfolder, 'Bennu-plots')
'''Plots of Bennu radiances as measured by OVIRS'''

val_and_test_path = Path('./validation_and_testing')
'''Validation results, each validation run saved in its own subdirectory'''
training_path = Path('./training')
'''Training results: network weights and loss history. Each training run in its own subdirectory'''

##########################################################################
# Gaussian distribution for noising generated data
mu = 0  # mean value for added noise
sigma = 0.0001  # standard deviation of noise distribution

##########################################################################
# Constraints for modeled radiances
# d_S_min, d_S_max = 0.7, 2.8  # Approximate heliocentric distance for asteroids where the problem is relevant, in AU
d_S_min, d_S_max = 0.8968944004459729 - 0.1, 1.355887651343651 + 0.1  # Heliocentric distances for Bennu, in AU:
# values from https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=bennu, with added margin of 0.1 AU
T_min, T_max = 150, 441  # Asteroid surface temperature range, in Kelvins. Min is arbitrary, max from subsolar
# temperature of ideal blackbody placed at perihelion (min d_s)
i_min, i_max = 0, 89  # Incidence angle, angle between surface normal and incident light, in degrees
e_min, e_max = 0, 89  # Emission angle, angle between surface normal and observer direction, in degrees
# IN TRUTH both emission and incidence angles can go up to 90... but if both hit 90, we get division by zero when
# calculating reflected radiance, and everything explodes

emissivity_min, emissivity_max = 0.2, 0.99  # Emissivity, ratio between thermal emission from body and from ideal bb:
# this can also represent some beaming effects and other angle dependencies!
p_min, p_max = 0.01, 0.40  # Geometrical albedo, ratio of light reflected from asteroid and from Lambertian disc

##########################################################################
# Neural network architecture parameters
conv_filters = 128
conv_kernel = 4
encoder_start = 1024  # 800
encoder_node_relation = 0.50
encoder_stop = 4
learning_rate = 9e-5

##########################################################################
# Neural network training parameters
refl_test_partition = 0.1  # Part of reflectances to be used for test data
activation = 'relu'
batch_size = 32  # Size of minibatch in training
epochs = 1500
# Early stop:
min_delta = 0.0005
patience = 50

##########################################################################
# Paths and filenames for saving trained networks
training_run_name = f'{epochs}epochs_{encoder_start}start_{encoder_stop}stop_{learning_rate}lr'
training_run_path = Path(training_path, training_run_name)
if os.path.isdir(training_run_path) == False:
    os.mkdir(training_run_path)  # Create directory for saving all the thing related to a training run
weights_path = Path(training_run_path, 'weights')
if os.path.isdir(weights_path) == False:
    os.mkdir(weights_path)
training_history_path = Path(training_run_path, f'{training_run_name}_trainHistory')

##########################################################################
# Path for saving results of hyperparameter tuning
hyperparameter_path = 'hyperparameter_tuning'  # KerasTuner wants the path as a string

##########################################################################
# Plot parameters, using default pyplot colors: '#1f77b4', '#ff7f0e', '#2ca02c'
uncor_plot_color = '#1f77b4'  # Blue
NNcor_plot_color = '#ff7f0e'  # Orange
ground_plot_color = '#2ca02c'  # Green

ideal_result_line_color = 'r'
mean_std_temp_color = 'k'

scatter_alpha = 0.02
scatter_marker = 'o'

# Limits suitable for plotting both synth and Bennu data from Bennu ground temp range
temperature_plot_ylim = (140, 380)
reflectance_mae_plot_ylim = (0, 0.02)
reflectance_sam_plot_ylim = (0.9997, 1)
reflrad_mae_plot_ylim = (0, 0.007)
reflrad_sam_plot_ylim = (0.999996, 1)

# # No limits, for plotting the whole synthetic data temperature range
# temperature_plot_ylim = (0, 0)
# reflectance_mae_plot_ylim = (0, 0)
# reflectance_sam_plot_ylim = (0, 0)
# reflrad_mae_plot_ylim = (0, 0)
# reflrad_sam_plot_ylim = (0, 0)

