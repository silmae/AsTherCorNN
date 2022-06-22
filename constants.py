"""
This file houses all the constants, including the physical constants and paths, and such.
"""

import os
from pathlib import Path
import numpy as np
import time
import scipy.constants

# Physical constants
c = scipy.constants.c  # 2.998e8  # speed of light in vacuum, m / s
kB = scipy.constants.Boltzmann  # 1.381e-23  # Boltzmann constant, m² kg / s² / K (= J / K)
h = scipy.constants.h  # 6.626e-34  # Planck constant, m² kg / s (= J s)
stefan_boltzmann = scipy.constants.Stefan_Boltzmann  # 5.67e-8 W / m² / K⁴, Stefan-Boltzmann constant
# emittance = 0.9  # Emittance of an asteroid, an approximation. Use Kirchoff's law (eps = 1-R) to get emittance from reflectance

# Wavelength vector
# step = 0.002  # µm
# wavelengths = np.arange(1, 2.5 + step, step=step)
step = 0.01  # µm
wavelengths = np.arange(0.46, 2.45 + step, step=step)  # TODO Should run from 0.45 to 2.45, why doesn't it?

# Paths

spectral_path = Path('./spectral_data')
Penttila_orig_path = Path('./spectral_data/reflectances/Penttila_asteroid_spectra/MyVISNIR-final-sampled-collection.dat')
'''Reflectances of asteroids'''
Penttila_aug_path = Path('./spectral_data/reflectances/Penttila_asteroid_spectra/MyVISNIR-simulated-simplified-taxonomy.dat')
'''Reflectance spectra of asteroids, augmented'''
albedo_path = Path('./spectral_data/reflectances/Penttila_asteroid_spectra/class-mean-albedos.tab')
'''Mean albedos of asteroid spectral classes'''
solar_path = Path('./spectral_data/solar-spectral-irradiance/solar_spectrum.txt')
'''Solar irradiance spectrum'''
rad_bunch_test_path = Path('./spectral_data/rad_bunch_test_bennu_random_no-noise_min-150K')
'''All synthetic test radiances, saved as a pickle'''
rad_bunch_training_path = Path('./spectral_data/rad_bunch_training_bennu_random_no-noise_min-150K')
'''All training data, saved as a pickle'''
radiance_path = Path('./spectral_data/radiances')
radiance_training_path = Path(radiance_path, 'training')
radiance_test_path = Path(radiance_path, 'test')

figfolder = Path('./figs')
refl_plots_path = Path(figfolder, 'asteroid-reflectance-plots')
rad_plots_path = Path(figfolder, 'radiance-plots')
max_temp_plots_path = Path(figfolder, 'max_temp_plots')
bennu_plots_path = Path(figfolder, 'Bennu-plots')
val_and_test_path = Path('./validation_and_testing')

training_path = Path('./training')

# Keys for variables
wl_key = 'wavelength'
R_key = 'reflectance'

# Gaussian distribution for noising
mu = 0  # mean
sigma = 0.0001 #0.02  # standard deviation

# Constraints for modeled radiances
# d_S_min, d_S_max = 0.7, 2.8  # Heliocentric distance for asteroids where the problem is relevant, in AU
d_S_min, d_S_max = 0.8968944004459729 - 0.1, 1.355887651343651 + 0.1  # Heliocentric distances for Bennu, in AU
T_min, T_max = 150, 430  # Asteroid surface temperature, in Kelvins
i_min, i_max = 0, 89  # Incidence angle, angle between surface normal and incident light, in degrees
e_min, e_max = 0, 89  # Emission angle, angle between surface normal and observer direction, in degrees
# IN TRUTH both emission and incidence angles can go up to 90... but if both hit 90, we get division by zero when
# calculating reflected radiance, and everything explodes

emissivity_min, emissivity_max = 0.2, 0.99  # Emissivity, ratio between thermal emission from body and from ideal bb. NB: Can also represent some beaming effects and such!
p_min, p_max = 0.01, 0.40  # Geometrical albedo, ratio of light reflected from asteroid and from Lambertian disc

# Neural network parameters
refl_test_partition = 0.1  # Part of reflectances to be used for test data
activation = 'relu'
batches = 16
epochs = 550

conv_filters = 128
conv_kernel = 4
encoder_start = 1024  # 800
encoder_node_relation = 0.50
encoder_stop = 4
learning_rate = 2e-5

training_run_name = f'{epochs}epochs_{encoder_start}start_{encoder_stop}stop_{learning_rate}lr'
training_run_path = Path(training_path, training_run_name)
if os.path.isdir(training_run_path) == False:
    os.mkdir(training_run_path)  # Create directory for saving all the thing related to a training run
weights_path = Path(training_run_path, 'weights')
if os.path.isdir(weights_path) == False:
    os.mkdir(weights_path)
training_history_path = Path(training_run_path, f'{training_run_name}_trainHistory')
# Early stop:
min_delta = 0.0001
patience = 50

# Paths for saving results of hyperparameter tuning
hyperparameter_path = 'hyperparameter_tuning'  # KerasTuner wants the path as a string

# Plot parameters, using default pyplot colors: '#1f77b4', '#ff7f0e', '#2ca02c'
uncor_plot_color = '#1f77b4'  # Blue
NNcor_plot_color = '#ff7f0e'  # Orange
ground_plot_color = '#2ca02c' # Green


