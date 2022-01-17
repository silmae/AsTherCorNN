import numpy as np
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path

import constants as C
import toml_handler as tomler
import solar as sol

def bb_radiance(T: float, eps: float, wavelength: np.ndarray):
    """
    Calculate and return approximate thermal emission (blackbody, bb) radiance spectrum using Planck's law. Angle
    dependence of emitted radiance is approximated as Lambertian. TODO If this does not work with OREX, change Lambert?

    :param T: float.
        Surface temperature, in Kelvins
    :param eps: float.
        Emissivity = sample emission spectrum divided by ideal bb spectrum of same temperature. Maybe in future accepts
        a vector, but only constants for now
    :param wavelength:
        vector of floats. Wavelengths where the emission is to be calculated, in micrometers

    :return L_th:
        vector of floats. Spectral radiance emitted by the surface.
    """

    # Define constants
    c = C.c  # speed of light in vacuum, m / s
    kB = C.kB  # Boltzmann constant, m² kg / s² / K (= J / K)
    h = C.h  # Planck constant, m² kg / s (= J s)

    L_th = np.zeros((len(wavelength),2))
    L_th[:, 0] = wavelength

    for i in range(len(wavelength)):
        wl = wavelength[i] / 1e6  # Convert wavelength from micrometers to meters
        L_th[i, 1] = eps * (2 * h * c**2) / ((wl**5) * (np.exp((h * c)/(wl * kB * T)) - 1))  # Apply Planck's law
        L_th[i, 1] = L_th[i,1] / 1e6  # Convert radiance from (W / m² / sr / m) to (W / m² / sr / µm)

    return L_th


def reflected_radiance(reflectance: np.ndarray, irradiance: np.ndarray, incidence_angle: float, emission_angle: float):
    """
    Calculate spectral radiance reflected from a surface, based on the surface reflectance, irradiance incident on it,
    and the phase angle of the measurement. Angle dependence is calculated using the Lommel-Seeliger model.

    :param reflectance: vector of floats
        Spectral reflectance.
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

    # Reflected radiance from flat surface with 0 phase angle
    reflrad[:, 1] = irradiance[:, 1] * reflectance

    # Angle dependence with L-S
    reflrad[:, 1] = (reflrad[:, 1] / (4 * np.pi)) * (1 / (np.cos(np.deg2rad(incidence_angle)) + np.cos(np.deg2rad(emission_angle))))

    return reflrad


def radiance2reflectance(radiance, d_S, phi, theta):
    insolation = sol.solar_irradiance(d_S, C.wavelengths)
    radiance = radiance / np.cos(np.deg2rad(theta))
    reflectance = radiance / ((insolation[:, 1] * np.cos(np.deg2rad(phi))) / np.pi)
    return reflectance

def noising(rad_data):
    """
    Apply Gaussian noise to spectral data

    :param rad_data: ndarray

        Radiance data to which the noise will be applied: first column is wavelength, second is spectral radiance

    :return:
        Noisified radiance data
    """
    mu = C.mu  # mean and standard deviation, defined with other constants
    sigma = C.sigma

    s = np.random.default_rng().normal(mu, sigma, len(rad_data))

    rad_data[:, 1] = rad_data[:, 1] + s

    return rad_data


def observed_radiance(d_S: float, incidence_ang: float, emission_ang: float, T: float, reflectance: np.ndarray, waves: np.ndarray, filename: str, test: bool, plots=False):

    # Calculate insolation at heliocentric distance of d_S
    insolation = sol.solar_irradiance(d_S, waves)

    # Calculate theoretical radiance reflected from an asteroid toward observer
    reflrad = reflected_radiance(reflectance, insolation, incidence_ang, emission_ang)

    # Calculate theoretical thermal emission from an asteroid's surface
    eps = C.emittance
    thermrad = bb_radiance(T, eps, waves)

    # Sum the two calculated spectral radiances
    sumrad = np.zeros((len(waves), 2))
    sumrad[:, 0] = waves
    sumrad[:, 1] = reflrad[:, 1] + thermrad[:, 1]

    # Applying noise to the summed data
    sumrad = noising(sumrad)

    # Collect the data into a dict
    rad_dict = {}
    meta = {'heliocentric_distance': d_S, 'incidence_angle': incidence_ang, 'emission_angle': emission_ang, 'surface_temperature': T,
            'emittance': eps}
    rad_dict['metadata'] = meta
    rad_dict['wavelength'] = waves
    rad_dict['reflected_radiance'] = reflrad[:, 1]
    rad_dict['emitted_radiance'] = thermrad[:, 1]
    rad_dict['sum_radiance'] = sumrad[:, 1]

    # Save the dict as .toml
    tomler.save_radiances(rad_dict, filename, test)

    if plots == True:

        # Plotting reflectance and radiances, saving as .png
        figfolder = C.figfolder

        plt.figure()
        plt.plot(C.wavelengths, reflectance)
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Reflectance')
        figpath = figfolder.joinpath(filename + '_reflectance.png')
        plt.savefig(figpath)

        plt.figure()
        plt.title(f'Radiances: d_S = {d_S}, i = {incidence_ang}, e = {emission_ang}, T = {T}')
        plt.plot(reflrad[:, 0], reflrad[:,1])  # Reflected radiance
        plt.plot(thermrad[:, 0], thermrad[:, 1])  # Thermally emitted radiance
        plt.plot(waves, reflrad[:, 1] + thermrad[:, 1])  # Sum of the two radiances
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Radiance')
        plt.legend(('Reflected', 'Thermal', 'Sum'))
        figpath = figfolder.joinpath(filename + ('_radiances.png'))
        plt.savefig(figpath)
        # plt.show()


def calculate_radiances(reflectance_list: list, test: bool):

    waves = C.wavelengths
    j = 1

    # From each reflectance, create 10 radiances calculated with different parameters
    for reflectance in reflectance_list:
        for i in range(10):
            # Create random variables from min-max ranges given in constants
            d_S = random.random() * (C.d_S_max - C.d_S_min) + C.d_S_min
            incidence_ang = random.randint(C.i_min, C.i_max)
            emission_ang = random.randint(C.e_min, C.e_max)
            T = random.randint(C.T_min, C.T_max)

            if j % 1000 == 0:
                # Calculate radiances with the given parameters and
                # save plots for every hundredth radiance and reflectance
                observed_radiance(d_S, incidence_ang, emission_ang, T, reflectance, waves, 'rads_' + str(j), test, plots=True)
            else:
                observed_radiance(d_S, incidence_ang, emission_ang, T, reflectance, waves, 'rads_' + str(j), test)

            j = j+1


def read_radiances(test: bool):
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
        rad_dict = tomler.read_radiance(filename, test)
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
