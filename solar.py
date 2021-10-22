from pathlib import Path
from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def solar_illuminance(distance):
    mat_path = Path('./D65_A_xyz.mat') # TODO nää polut kans mieluummin kootusti jonnekin yhteen paikkaan
    mat = io.loadmat(mat_path)
    xyz = mat['xyz']
    V_lambda = xyz[:, [0, 2]]

    sol_path = Path('./solar_spectrum.txt')
    solar = pd.read_table(sol_path).values
    # Convert from µm to nm, and 1/µm to 1/nm
    solar[:, 0] = solar[:, 0] * 1000
    solar[:, 1] = solar[:, 1] / 1000

    solar_interp = np.interp(V_lambda[:, 0], solar[:, 0], solar[:, 1]) / distance**2

    plt.plot(V_lambda[:, 0], solar_interp)
    # plt.show()

    solar_V_lambda = solar_interp * V_lambda[:, 1]
    plt.plot(V_lambda[:, 0], solar_V_lambda)
    plt.xlabel('Wavelength [nm]')
    # plt.ylabel('Irradiance [W/m^2/nm]')
    plt.legend(('Irradiance [W/m^2/nm]' , 'Scaled to V-lambda'))
    plt.show()

    # Convert to lm/m^2/nm
    solar_spectral_illuminance = solar_V_lambda * 683

    # Integrate (sum) over wl:s to get total illuminance in lux
    solar_illuminance = sum(solar_spectral_illuminance)

    return(solar_illuminance)


def solar_irradiance(distance):
    """
    Calculate solar spectral irradiance at a specified heliocentric distance.
    Irradiance data from NREL: https://www.nrel.gov/grid/solar-resource/spectra-astm-e490.html
    :param distance: float. Heliocentric distance in astronomical units
    :return: wavelength vector (in nanometers) and spectral irradiance in one array
    """
    sol_path = Path('./solar_spectrum.txt')  # A collection of channels from 1 to 4 µm saved into a txt file
    solar = pd.read_table(sol_path).values

    # Convert from µm to nm, and 1/µm to 1/nm. Comment these away is working with IR wavelengths
    # solar[:, 0] = solar[:, 0] * 1000
    # solar[:, 1] = solar[:, 1] / 1000

    # Scale with heliocentric distance, using the inverse square law
    solar[:,1] = solar[:,1] / distance**2

    # plt.plot(solar[:,0], solar[:,1])
    # plt.xlabel('Wavelength [µm]')
    # plt.ylabel('Irradiance [W/m^2/µm]')
    # plt.show()

    return solar

