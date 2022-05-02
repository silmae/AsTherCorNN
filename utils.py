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

    If no albedo, emissivity, and beaming parameter give as arguments, method calculates the blackbody radiative
    equilibrium temperature at the given heliocentric distance.

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


def maximum_temperatures(distance_min: float = 0.5, distance_max: float = 4.0, num: int = 50):
    """
    Calculate and plot maximum temperatures of ideal blackbody radiators warmed by the Sun, placed at different
    heliocentric distances.

    :param distance_min: float
        Minimum heliocentric distance in astronomical units, default is 0.5 AU
    :param distance_max: float
        Maximum heliocentric distance in astronomical units, default is 4.0 AU
    :param num: int
        Number of temperatures to be calculated, default is 50

    :return ss_temps_max: list
        A list of maximum subsolar temperatures
    """

    d_S = np.linspace(distance_min, distance_max, num=num)
    ss_temps_max = []

    for distance in d_S:
        temperature_max = calculate_subsolar_temperature(distance)
        ss_temps_max.append(temperature_max)

    plt.figure()
    plt.plot(d_S, ss_temps_max)
    plt.xlabel('Heliocentric distance [AU]')
    plt.ylabel('Subsolar temperature [K]')
    plt.grid()
    plt.savefig(Path(C.max_temp_plots_path, 'ss-temp_hc-dist.png'))
    # plt.show()

    return ss_temps_max


def thermal_error_from_hc_distance(distance_min: float, distance_max: float, samples: int):
    """
    Calculate and plot error in reflectance at 2.45 µm caused by thermal emission, at different heliocentric distances.
    Result can be used to constrain the maximum heliocentric distance where this problem is significant enough to merit
    correction.

    :param distance_min: float
        Minimum heliocentric distance in astronomical units
    :param distance_max: float
        Maximum heliocentric distance in astronomical units
    :param samples:
        Number of samples to be calculated
    """
    # FINDING BOUNDARY CONDITIONS FOR HELIOCENTRIC DISTANCE
    # Looking at a worst-case scenario: dark asteroid (T class, p = 0.035), temperature according to subsolar temp of
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

            # Un-normalize reflectance by scaling it with visual geometrical albedo
            spectral_reflectance = norm_reflectance * geom_albedo
            # Convert reflectance to single-scattering albedo, using Lommel-Seeliger
            spectral_single_scattering_albedo = 8 * spectral_reflectance

            # Print if the physical limits of min and max reflectance are exceeded
            if np.max(spectral_single_scattering_albedo) > 1 or np.min(spectral_single_scattering_albedo) < 0:
                print(f'Unphysical reflectance detected! Max {np.max(spectral_single_scattering_albedo)}, min {np.min(spectral_single_scattering_albedo)}')

    # A list of heliocentric distances
    distances = np.linspace(distance_min, distance_max, samples)
    # A list of temperatures
    temperatures = maximum_temperatures(distance_min, distance_max, samples)
    # Empty list for storing errors
    errors = []

    i = 0
    for distance in distances:
        insolation = solar_irradiance(distance, C.wavelengths)
        radiance_dict = rad.observed_radiance(d_S=distance,
                                              incidence_ang=0,
                                              emission_ang=0,
                                              T=temperatures[i],
                                              reflectance=spectral_single_scattering_albedo,
                                              waves=C.wavelengths,
                                              constant_emissivity=True,
                                              filename='filename',
                                              test=True,
                                              save_file=False)
        reflected_radiance = radiance_dict['reflected_radiance']
        # thermal_radiance = radiance_dict['emitted_radiance']
        sum_radiance = radiance_dict['sum_radiance']

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
        plt.savefig(Path(C.max_temp_plots_path, f'{round(distance, 2)}-AU_{round(temperatures[i], 2)}-K_radiance.png'))
        plt.close(fig)

        fig = plt.figure()
        plt.plot(C.wavelengths, reference_reflectance)
        plt.plot(C.wavelengths, test_reflectance)
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Normalized reflectance')
        plt.legend(('Reference', 'Test'))
        plt.savefig(Path(C.max_temp_plots_path, f'{round(distance, 2)}-AU_{round(temperatures[i], 2)}-K_reflectance.png'))
        plt.close(fig)

        i = i + 1

    # Plot error as function of heliocentric distance
    plt.figure()
    plt.plot(distances, errors)
    plt.xlabel('Heliocentric distance [AU]')
    plt.ylabel('Reflectance error at 2.5 µm [%]')
    plt.yscale('log')
    plt.grid()
    plt.savefig(Path(C.max_temp_plots_path, f'{i}_error_hc-distance.png'))
    plt.show()


def solar_irradiance(distance, wavelengths, plot=False):
    """
    Calculate solar spectral irradiance at a specified heliocentric distance, interpolated to match wl-vector.
    Solar spectral irradiance data at 1 AU outside the atmosphere was taken from from NREL:
    https://www.nrel.gov/grid/solar-resource/spectra-astm-e490.html

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

    # # Convert from µm to nm, and 1/µm to 1/nm. Comment these away if working with IR wavelengths
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
    Which radiance to plot, between 0 and 6599. Default is 22, where thermal error could be seen clearly for one data
    generation cycle (possibly not the case anymore).
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