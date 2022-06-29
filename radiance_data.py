"""
Methods for working with spectral radiances: calculating thermal radiance and reflected radiance, and simulating
observed radiance as their sum. Calculating normalized reflectance from radiance. Wrapper for calculating a dataset of
several radiances from reflectances.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path

import constants as C
import file_handling as FH
import utils


def thermal_radiance(T: float, emissivity: float or list or np.ndarray, wavelength: np.ndarray):
    """
    Calculate and return approximate thermal emission (blackbody, bb) radiance spectrum using Planck's law. Angle
    dependence of emitted radiance is approximated as Lambertian. TODO If this does not work with OREX, change Lambert?

    :param T:
        Surface temperature, in Kelvins
    :param emissivity:
        Emissivity = sample emission spectrum divided by ideal blackbody spectrum of same temperature. Float or
        vector/list with same number of elements as reflectance.
    :param wavelength:
        vector of floats. Wavelengths where the emission is to be calculated, in micrometers

    :return L_th:
        vector of floats. Spectral radiance emitted by the surface.
    """

    # Define constants
    c = C.c  # speed of light in vacuum, m / s
    kB = C.kB  # Boltzmann constant, m² kg / s² / K (= J / K)
    h = C.h  # Planck constant, m² kg / s (= J s)

    if type(emissivity) == float or type(emissivity) == np.float64 or type(emissivity) == np.float32 or len(emissivity) == 1:
        # If a single float, make it into a vector where each element is that number
        eps = np.empty((len(wavelength), 1))
        eps.fill(emissivity)
    elif type(emissivity) == list:
        # If emissivity is a list with correct length, convert to ndarray
        if len(emissivity) == len(C.wavelengths):
            eps = np.asarray(emissivity)
        else:
            print('Emissivity list was not same length as wavelength vector. Stopping execution...')
            quit()
    elif type(emissivity) == np.ndarray:
        # If emissivity array is of correct shape, rename it to emittance and proceed
        if emissivity.shape == C.wavelengths.shape or emissivity.shape == (
        C.wavelengths.shape, 1) or emissivity.shape == (1, C.wavelengths.shape):
            eps = emissivity
        else:
            print('Emissivity array was not same shape as wavelength vector. Stopping execution...')
            quit()

    L_th = np.zeros((len(wavelength), 2))
    L_th[:, 0] = wavelength

    for i in range(len(wavelength)):
        wl = wavelength[i] / 1e6  # Convert wavelength from micrometers to meters
        L_th[i, 1] = eps[i] * (2 * h * c ** 2) / ((wl ** 5) * (np.exp((h * c) / (wl * kB * T)) - 1))  # Apply Planck's law
        L_th[i, 1] = L_th[i, 1] / 1e6  # Convert radiance from (W / m² / sr / m) to (W / m² / sr / µm)

    return L_th


def reflected_radiance(reflectance: np.ndarray, irradiance: np.ndarray, incidence_angle: float, emission_angle: float):
    """
    Calculate spectral radiance reflected from a surface, based on the surface reflectance, irradiance incident on it,
    and the phase angle of the measurement. Angle dependence (BRDF) is calculated using the Lommel-Seeliger model.

    :param reflectance: vector of floats
        Spectral reflectance, calculated using estimation for single-scattering albedo
    :param irradiance: vector of floats
        Spectral irradiance incident on the surface.
    :param incidence_angle: float
        Incidence angle of light in degrees, measured from surface normal
    :param emission_angle: float
        Angle between surface normal and observer direction, in degrees

    :return: vector of floats.
        Spectral radiance reflected toward the observer.
    """

    wavelength = irradiance[:, 0]
    reflrad = np.zeros((len(wavelength), 2))
    reflrad[:, 0] = wavelength

    # BRDF with Lommel-Seeliger
    reflrad[:, 1] = (reflectance / (4 * np.pi)) * (1 / (np.cos(np.deg2rad(incidence_angle)) + np.cos(np.deg2rad(emission_angle))))

    # Reflected radiance from incident irradiance and BRDF
    reflrad[:, 1] = irradiance[:, 1] * reflrad[:, 1]

    return reflrad


def radiance2norm_reflectance(radiance):
    """
    Calculates a spectrum of normalized reflectance from a spectrum of reflected radiance. Divides the radiance
    by the solar spectral irradiance at heliocentric distance of 1 AU, and normalizes the resulting reflectance
    spectrum so that reflectance is 1 at the wavelength of 0.55 micrometers (a common convention with asteroid
    reflectance spectra).

    :param radiance: ndarray
        Spectral radiance
    :return: norm_reflectance: ndarray
        Normalized reflectance
    """

    # Insolation at 1.0, the heliocentric distance does not matter with normalized data
    insolation = utils.solar_irradiance(1.0, C.wavelengths)
    reflectance = radiance / insolation[:, 1]

    # Find the index where wavelength is closest to 0.55 µm
    array = np.asarray(C.wavelengths)
    idx = (np.abs(array - 0.55)).argmin()

    # Normalization to R(0.55 µm) = 1
    norm_reflectance = reflectance / reflectance.squeeze()[idx]

    return norm_reflectance


def noising(rad_data, mu, sigma):
    """
    Apply Gaussian noise to spectral data

    :param rad_data: ndarray
        Radiance data to which the noise will be applied: first column is wavelength, second is spectral radiance
    :param mu:
        Mean value of Gaussian distribution from which the noise is generated
    :param sigma:
        Standard deviation of Gaussian distribution from which the noise is generated
    :return:
        Noisified radiance data
    """

    # Generate noise: pull a random value from Gaussian distribution for every element of the radiance vector
    s = np.random.default_rng().normal(mu, sigma, len(rad_data))

    # Add noise to data
    rad_data[:, 1] = rad_data[:, 1] + s

    return rad_data


def observed_radiance(d_S: float, incidence_ang: float, emission_ang: float, T: float, reflectance: np.ndarray,
                      waves: np.ndarray, emissivity: float or list or np.ndarray, filename: str, test: bool,
                      plots=False, save_file=True):
    """
    Simulate observed radiance with given parameters. Calculates reflected and thermally emitted radiances in separate
    functions, and sums them to get observed radiance. Adds noise to the summed radiance. Saves separate and summed
    radiances to a .toml file together with observation related metadata provided in function arguments. Also saves
    plots of reflectance and radiances, if specified in arguments.

    :param d_S: float
        Heliocentric distance in astronomical units
    :param incidence_ang: float
        Incidence angle in degrees
    :param emission_ang: float
        Emission angle in degrees
    :param T: float
        Surface temperature in Kelvin
    :param reflectance: ndarray
        Spectral reflectance
    :param waves: ndarray
        Wavelength vector in micrometers
    :param emissivity: float, list, or ndarray
        If float, assumed to be constant over wavelengths. If ndarray, assumed to have same wavelength vector as defined
        in constants.py
    :param filename: string
        Name which will be included in files related to the radiances (plots and .toml), without extension
    :param test: boolean
        Whether the simulated measurement will be used for testing or training, only affects save location on disc
    :param plots: boolean
        Whether plots will be made for this simulated measurement
    """
    # Calculate insolation at heliocentric distance of d_S
    insolation = utils.solar_irradiance(d_S, waves)

    # Calculate theoretical radiance reflected from an asteroid toward observer
    reflrad = reflected_radiance(reflectance, insolation, incidence_ang, emission_ang)

    # Calculate theoretical thermal emission from an asteroid's surface
    thermrad = thermal_radiance(T, emissivity, waves)

    # Sum the two calculated spectral radiances
    sumrad = np.zeros((len(waves), 2))
    sumrad[:, 0] = waves
    sumrad[:, 1] = reflrad[:, 1] + thermrad[:, 1]

    # Applying noise to the summed data
    sumrad = noising(sumrad, C.mu, C.sigma)

    # Collect the data into a dict
    rad_dict = {}
    meta = {'heliocentric_distance': d_S, 'incidence_angle': incidence_ang, 'emission_angle': emission_ang, 'surface_temperature': T,
            'emissivity': emissivity}
    rad_dict['metadata'] = meta
    rad_dict['wavelength'] = waves
    rad_dict['reflected_radiance'] = reflrad[:, 1]
    rad_dict['emitted_radiance'] = thermrad[:, 1]
    rad_dict['sum_radiance'] = sumrad[:, 1]

    # Save the dict as .toml
    if save_file == True:
        FH.save_radiances(rad_dict, filename, test)

    if plots == True:

        # Plotting reflectance and radiances, saving as .png
        figfolder = C.figfolder

        fig = plt.figure()
        plt.plot(C.wavelengths, reflectance)
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Reflectance')
        figpath = figfolder.joinpath(filename + '_reflectance.png')
        plt.savefig(figpath)
        plt.close(fig)

        fig = plt.figure()
        plt.title(f'Radiances: d_S = {d_S}, i = {incidence_ang}, e = {emission_ang}, T = {T}')
        plt.plot(reflrad[:, 0], reflrad[:,1])  # Reflected radiance
        plt.plot(thermrad[:, 0], thermrad[:, 1])  # Thermally emitted radiance
        plt.plot(waves, reflrad[:, 1] + thermrad[:, 1])  # Sum of the two radiances
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Radiance')
        plt.legend(('Reflected', 'Thermal', 'Sum'))
        figpath = Path(C.rad_plots_path, f'{filename}_radiances.png')
        plt.savefig(figpath)
        plt.close(fig)
        # plt.show()

    return rad_dict


def calculate_radiances(reflectance_list: list, test: bool, samples_per_temperature: int = 200, emissivity_type: str = 'constant'):
    """
    Generate vector of temperature values based on minimum given in constants and maximum calculated from minimum
    heliocentric distance. For each temperature simulate a number of observed radiances with reflectance vector
    chosen randomly. Values for incidence- and emission angles and heliocentric distance are likewise random, with
    constraints for their values pulled from constants.py

    Each set of observed radiances is saved into its own .toml file, every 10 000th radiance and its reflectance is
    plotted. Radiances are returned without their metadata, as ndarray.

    :param reflectance_list: list
        Spectral reflectances from which the radiances will be created.
    :param test: boolean
        Whether the data will be used for testing or training. Affects only the save location, calculation is identical.
    :param samples_per_temperature: int
        How many spectra will be generated per temperature value. Default is 200.
    :param emissivity_type: string
        Must be "constant", "kirchhoff", or "random". First is 0.9 for all samples, second is spectral emissivity
        calculated from spectral reflectance with Kirchhoff's law, third is random value between min and max given
        in constants.py (stays constant over wavelengths)

    :return: summed, separate: ndarrays
        Arrays containing summed spectra and separate reflected and thermal spectra

    *:raises: ValueError
        If emissivity_type does not correspond to allowed values.
    """

    waves = C.wavelengths

    # Minimum temperature pulled from constants
    T_min = int(C.T_min)
    # Maximum temperature as theoretical maximum at minimum heliocentric distance
    T_max = int(utils.calculate_subsolar_temperature(C.d_S_min))

    # Generate vector of temperatures from minimum to maximum with 1 K separation
    temperature_vector = np.linspace(T_min, T_max, T_max - T_min + 1)

    # Empty arrays for storing data vectors
    length = len(waves)
    samples = len(temperature_vector) * samples_per_temperature
    sum_radiances = np.zeros((samples, length))
    temperatures = np.zeros((samples, 1))
    emissivities = np.zeros((samples, 1))
    
    # reflected = np.zeros((samples, length))
    # therm = np.zeros((samples, length))

    j = 0

    # Calculate radiances for each temperature
    for temperature in temperature_vector:
        for i in range(samples_per_temperature):
            # Create random variables from min-max ranges given in constants
            d_S = random.random() * (C.d_S_max - C.d_S_min) + C.d_S_min
            incidence_ang = random.randint(C.i_min, C.i_max)
            emission_ang = random.randint(C.e_min, C.e_max)

            # Take a random reflectance from the list given as argument
            reflectance_index = random.randint(0, len(reflectance_list)-1)
            reflectance = reflectance_list[reflectance_index]

            # Emissivity according to type given in arguments
            if emissivity_type == 'kirchhoff':
                emissivity = 1 - reflectance
            elif emissivity_type == 'constant':
                emissivity = 0.9
            elif emissivity_type == 'random':
                emissivity = random.uniform(C.emissivity_min, C.emissivity_max)
            else:
                raise ValueError(f"Parameter 'emissivity_type' value not valid: {emissivity_type}. Try 'constant', 'kirchhoff', or 'random'.")

            if j % 10000 == 0:
                # Calculate radiances with the given parameters and
                # save plots for every 10 000th radiance and reflectance
                obs_rad_dict = observed_radiance(d_S, incidence_ang, emission_ang, temperature, reflectance, waves, emissivity, 'rads_' + str(j), test, plots=True)
            else:
                obs_rad_dict = observed_radiance(d_S, incidence_ang, emission_ang, temperature, reflectance, waves, emissivity, 'rads_' + str(j), test)

            # Store summed radiance, temperature, and emittance into arrays, discard other data in the dict
            sum_radiances[j, :] = obs_rad_dict['sum_radiance']
            temperatures[j, :] = temperature
            emissivities[j, :] = emissivity

            j = j+1

        # Place reflected and thermal spectra into one array for returning
        thermal_parameters = np.zeros((samples, 2))
        thermal_parameters[:, 0] = temperatures.flatten()
        thermal_parameters[:, 1] = emissivities.flatten()

    return sum_radiances, thermal_parameters


def read_radiances(test: bool):
    """
    Read .toml files in folders for test and training data, place radiances in ndarrays and discard metadata.

    :param test:
        Whether data should be read from test or training folder
    :return:
        Array of summed radiances, array of separate radiances
    """
    if test == True:
        folder_path = C.radiance_test_path
    else:
        folder_path = C.radiance_training_path

    file_list = os.listdir(folder_path)
    # Shuffle the list of files, so they are not read in the order they were created: data next to each other are not
    # calculated from the same reflectance after shuffling
    random.shuffle(file_list)

    length = len(C.wavelengths)
    samples = len(file_list)
    summed = np.zeros((samples, length))
    reflected = np.zeros((samples, length))
    therm = np.zeros((samples, length))

    i = 0
    for filename in file_list:
        rad_dict = FH.read_radiance(filename, test)
        # Extract the radiances, discard the metadata
        summed[i, :] = rad_dict['sum_radiance']
        reflected[i, :] = rad_dict['reflected_radiance']
        therm[i, :] = rad_dict['emitted_radiance']
        print(f'Read file {i} out of {len(file_list)}')
        i = i + 1

    separate = np.zeros((samples, length, 2))
    separate[:, :, 0] = reflected
    separate[:, :, 1] = therm

    return summed, separate
