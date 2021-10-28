'''
This file has all the constants, including the physical ones and string names and paths, and such.
'''
from pathlib import Path
import numpy as np

# Physical constants
c = 2.998e8  # speed of light in vacuum, m / s
kB = 1.381e-23  # Boltzmann constant, m² kg / s² / K (= J / K)
h = 6.626e-34  # Planck constant, m² kg / s (= J s)
emittance = 0.9  # Emittance of an asteroid TODO Use Kirchoff's law (eps = 1-R) to get emittance from reflectance?

# Wavelength vector
step = 0.002  # µm
wavelengths = np.arange(1, 2.5 + step, step=step)

# Paths
Maturilli_path = Path('./spectral_data/reflectances/asteroid_analogues/refle/MIR')  # Reflectance spectra of asteroid analogues
Gaffey_path = Path('./spectral_data/reflectances/Gaffey_meteorite_spectra/data/spectra')  # Reflectance spectra of meteorites
figfolder = Path('./figs')
solar_path = Path('./spectral_data/solar_spectrum.txt')  # Solar irradiance spectrum
augmented_path = Path('./spectral_data/reflectances/augmented')  # A folder of augmented spectra
radiance_path = Path('./spectral_data/radiances')

# Keys for variables
wl_key = 'wavelength'
R_key = 'reflectance'

# Gaussian distribution for noising
mu = 0  # mean
sigma = 0.01  # standard deviation

# Constraints for modeled radiances
d_S_min, d_S_max = 0.7, 2  # Heliocentric distance, in AU
T_min, T_max = 200, 400  # Asteroid surface temperature, in Kelvins
phi_min, phi_max = 0, 80  # Measurement phase angle, in degrees
theta = 10  # Angle between surface normal and observer direction, in degrees


