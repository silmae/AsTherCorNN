from pathlib import Path
from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import constants as C
import file_handling as FH
import radiance_data as rad


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

    T_ss = (((1 - albedo) * 1361 * (1 / heliocentric_distance ** 2)) / (beaming_param * emissivity * C.stefan_boltzmann)) ** 0.25

    return T_ss


def plot_maximum_temperatures(distance_min: float = 0.5, distance_max: float = 4.0):
    """
    Calculate and plot maximum temperatures of ideal blackbody radiators warmed by the Sun, placed at different
    heliocentric distances.

    :param distance_min: float
        Minimum heliocentric distance in astronomical units, default is 0.5 AU
    :param distance_max: float
        Maximum heliocentric distance in astronomical units, default is 4.0 AU

    :return ss_temps_max: list
        A list of maximum subsolar temperatures
    """

    d_S = np.linspace(distance_min, distance_max)
    ss_temps_max = []

    for distance in d_S:
        temperature_max = calculate_subsolar_temperature(distance)
        ss_temps_max.append(temperature_max)

    plt.figure()
    plt.plot(d_S, ss_temps_max)
    plt.xlabel('Heliocentric distance [AU]')
    plt.ylabel('Subsolar temperature [K]')
    plt.savefig(Path(C.figfolder, 'ss-temp_hc-dist.png'))
    plt.show()
    return ss_temps_max


def solar_irradiance(distance, wavelengths, plot=False):
    """
    Calculate solar spectral irradiance at a specified heliocentric distance, interpolated to match wl-vector.
    Solar spectral irradiance data at 1 AU from NREL: https://www.nrel.gov/grid/solar-resource/spectra-astm-e490.html

    :param distance: float
        Heliocentric distance in astronomical units
    :param wavelengths: vector of floats
        Wavelength vector (in µm), to which the insolation will be interpolated
    :param plot: boolean
        Whether a plot will be shown of the calculated spectral irradiance

    :return: ndarray
        wavelength vector (in nanometers) and spectral irradiance in one ndarray
    """
    sol_path = C.solar_path  # A collection of channels from 0.45 to 2.50 µm saved into a txt file
    solar = pd.read_table(sol_path).values

    # Convert from µm to nm, and 1/µm to 1/nm. Comment these away if working with IR wavelengths
    # solar[:, 0] = solar[:, 0] * 1000
    # solar[:, 1] = solar[:, 1] / 1000

    # Scale with heliocentric distance, using the inverse square law
    solar[:, 1] = solar[:, 1] / distance**2

    # Interpolate to match the given wavelength vector
    interp_insolation = np.zeros((len(wavelengths), 2))
    interp_insolation[:, 0] = wavelengths
    interp_insolation[:, 1] = np.interp(wavelengths, solar[:, 0], solar[:, 1])

    if plot == True:
        plt.plot(solar[:, 0], solar[:, 1])
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Irradiance [W / m² / µm]')
        plt.show()

    return interp_insolation


def plot_example_radiance_reflectance(num: int = 22):
    """
    Load a set of test radiances from a toml file, plot reflected radiance and sum of reflected and thermal for
    comparison. Calculate normalized reflectance from both and also plot them.

    :param num: int
    Which radiance to plot, between 0 and 6599. Default is 22, where thermal error can be seen clearly.
    """
    rads = FH.load_toml(Path(C.radiance_test_path, f'rads_{num}.toml'))
    refrad = rads['reflected_radiance']
    sumrad = rads['sum_radiance']
    meta = rads['metadata']

    refR = rad.radiance2norm_reflectance(refrad)
    sumR = rad.radiance2norm_reflectance(sumrad)

    plt.figure()
    plt.plot(C.wavelengths, refrad)
    plt.plot(C.wavelengths, sumrad)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Radiance [W / m² / sr / µm]')
    plt.legend(('Reference', 'With thermal'))

    plt.figure()
    plt.plot(C.wavelengths, refR)
    plt.plot(C.wavelengths, sumR)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Normalized reflectance')
    plt.legend(('Reference', 'With thermal'))

    plt.show()