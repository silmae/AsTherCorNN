'''
This file has all the constants, including the physical constants and paths, and such.
'''
import os
from pathlib import Path
import numpy as np

# Physical constants
c = 2.998e8  # speed of light in vacuum, m / s
kB = 1.381e-23  # Boltzmann constant, m² kg / s² / K (= J / K)
h = 6.626e-34  # Planck constant, m² kg / s (= J s)
emittance = 0.9  # Emittance of an asteroid TODO Use Kirchoff's law (eps = 1-R) to get emittance from reflectance?

# Wavelength vector
# step = 0.002  # µm
# wavelengths = np.arange(1, 2.5 + step, step=step)
step = 0.01  # µm
wavelengths = np.arange(0.46, 2.45 + step, step=step)  # TODO Should run from 0.45 to 2.45, why doesn't it?

# Paths
Maturilli_path = Path('./spectral_data/reflectances/asteroid_analogues/refle/MIR')  # Reflectance spectra of asteroid analogues
Gaffey_path = Path('./spectral_data/reflectances/Gaffey_meteorite_spectra/data/spectra')  # Reflectance spectra of meteorites
Penttila_orig_path = Path('./spectral_data/reflectances/Penttila_asteroid_spectra/MyVISNIR-final-sampled-collection.dat')   # Reflectances of asteroids
Penttila_aug_path = Path('./spectral_data/reflectances/Penttila_asteroid_spectra/MyVISNIR-simulated-simplified-taxonomy.dat')  # Reflectance spectra of asteroids, augmented
albedo_path = Path('./spectral_data/reflectances/Penttila_asteroid_spectra/class-mean-albedos.tab')
figfolder = Path('./figs')
refl_plots_path = Path(figfolder, 'asteroid-reflectance-plots')
spectral_path = Path('./spectral_data')
solar_path = Path('./spectral_data/solar_spectrum.txt')  # Solar irradiance spectrum
augmented_path = Path('./spectral_data/reflectances/augmented')  # A folder of augmented spectra
augmented_training_path = Path(augmented_path, 'training')
augmented_test_path = Path(augmented_path, 'test')
radiance_path = Path('./spectral_data/radiances')
radiance_training_path = Path(radiance_path, 'training')
radiance_test_path = Path(radiance_path, 'test')
training_path = Path('./training')
spectral_path = Path('./spectral_data')
rad_bunch_test_path = Path('./spectral_data/rad_bunch_test')  # All radiances, saved as a dict
rad_bunch_training_path = Path('./spectral_data/rad_bunch_training')

# Keys for variables
wl_key = 'wavelength'
R_key = 'reflectance'

# Gaussian distribution for noising
mu = 0  # mean
sigma = 0.01  # standard deviation

# Constraints for modeled radiances
d_S_min, d_S_max = 0.7, 2  # Heliocentric distance, in AU
T_min, T_max = 200, 450  # Asteroid surface temperature, in Kelvins
i_min, i_max = 0, 89  # Measurement phase angle, in degrees
e_min, e_max = 0, 89  # Emission angle, angle between surface normal and observer direction, in degrees
# IN TRUTH both emission and incidence angles can go up to 90... but if both hit 90, we get division by zero when
# calculating reflected radiance, and everything explodes

# Neural network parameters
refl_test_partition = 0.1  # Part of reflectances to be used for test data
activation = 'relu'
learning_rate = 5e-6
batches = 32
epochs = 1000
waist = 32  # Autoencoder middle layer node count
training_run_name = f'{epochs}epochs_{waist}waist_{learning_rate}lr'
training_run_path = Path(training_path, training_run_name)
os.mkdir(training_run_path)  # Create directory for saving all the thing related to a training run
weights_path = Path(training_run_path, 'weights')
os.mkdir(weights_path)
training_history_path = Path(training_run_path, f'{training_run_name}_trainHistory')
# Early stop:
min_delta = 1
patience = 50


