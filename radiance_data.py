import numpy as np
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path

import constants as C
import toml_handler as tomler
import solar as sol

def bb_radiance(T: float, eps: float, theta: float, wavelength: np.ndarray):
    """
    Calculate and return approximate thermal emission (blackbody, bb) radiance spectrum using Planck's law. Angle
    dependence of emitted radiance is calculated with Lambert's cosine law.

    :param T: float.
        Surface temperature, in Kelvins
    :param eps: float.
        Emissivity = sample emission spectrum divided by ideal bb spectrum. Maybe in future accepts
        a vector, but only constants for now
    :param theta: float
        Angle between surface normal and observer direction (emission angle), in degrees
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
        L_th[i, 1] = L_th[i, 1] * np.cos(np.deg2rad(theta))  # Apply Lambert's cosine law
        L_th[i, 1] = L_th[i,1] / 1e6  # Convert radiance from (W / m² / sr / m) to (W / m² / sr / µm)

    L_th = noising(L_th)

    return L_th


def reflected_radiance(reflectance: np.ndarray, irradiance: np.ndarray, phi: float, theta: float):
    """
    Calculate spectral radiance reflected from a surface, based on the surface reflectance, irradiance incident on it,
    and the phase angle of the measurement. The surface normal is assumed to point towards the observer.

    :param reflectance: vector of floats
        Spectral reflectance.
    :param irradiance: vector of floats
        Spectral irradiance incident on the surface.
    :param phi: float
        Phase angle of the measurement, in degrees
    :param theta: float
        Angle between surface normal and observer direction, in degrees

    :return: vector of floats.
        Spectral radiance reflected toward the observer.
    """

    wavelength = irradiance[:, 0]
    reflrad = np.zeros((len(wavelength), 2))
    reflrad[:, 0] = wavelength

    reflrad[:, 1] = reflectance[:, 1] * (irradiance[:, 1] * np.cos(np.deg2rad(phi))) / np.pi
    reflrad[:, 1] = reflrad[:, 1] * np.cos(np.deg2rad(theta))

    # Applying noise to the data
    reflrad = noising(reflrad)

    return reflrad


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


def observed_radiance(d_S: float, phi: float, theta: float, T: float, reflectance: np.ndarray, waves: np.ndarray, filename: str, plots=False):

    # Calculate insolation at heliocentric distance of d_S
    insolation = sol.solar_irradiance(d_S, waves)

    # Calculate theoretical radiance reflected from an asteroid toward observer
    reflrad = reflected_radiance(reflectance, insolation, phi, theta)

    # Calculate theoretical thermal emission from an asteroid's surface
    eps = C.emittance
    thermrad = bb_radiance(T, eps, theta, waves)

    # Sum the two calculated spectral radiances
    sumrad = np.zeros((len(waves), 2))
    sumrad[:, 0] = waves
    sumrad[:, 1] = reflrad[:, 1] + thermrad[:, 1]

    # Collect the data into a dict
    rad_dict = {}
    meta = {'heliocentric_distance': d_S, 'phase_angle': phi, 'emission_angle': theta, 'surface_temperature': T,
            'emittance': eps}
    rad_dict['metadata'] = meta
    rad_dict['wavelength'] = waves
    rad_dict['reflected_radiance'] = reflrad[:, 1]
    rad_dict['emitted_radiance'] = thermrad[:, 1]
    rad_dict['sum_radiance'] = sumrad[:, 1]

    # Save the dict as .toml
    tomler.save_radiances(rad_dict, filename)

    if plots == True:

        # Plotting reflectance and radiances, saving as .png
        figfolder = C.figfolder

        plt.figure()
        plt.plot(reflectance[:, 0], reflectance[:, 1])
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Reflectance')
        figpath = figfolder.joinpath(filename + '_reflectance.png')
        plt.savefig(figpath)

        plt.figure()
        plt.title(f'Radiances: d_S = {d_S}, \phi = {phi}, T = {T}')
        plt.plot(reflrad[:, 0], reflrad[:,1])  # Reflected radiance
        plt.plot(thermrad[:, 0], thermrad[:, 1])  # Thermally emitted radiance
        plt.plot(waves, reflrad[:, 1] + thermrad[:, 1])  # Sum of the two radiances
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Radiance')
        plt.legend(('Reflected', 'Thermal', 'Sum'))
        figpath = figfolder.joinpath(filename + ('_radiances.png'))
        plt.savefig(figpath)
        # plt.show()

def calculate_radiances():

    waves = C.wavelengths
    theta = C.theta
    aug_list = os.listdir(C.augmented_path)

    j = 1
    for filename in aug_list:
        aug_filepath = C.augmented_path.joinpath(filename)
        reflectance = tomler.read_aug_reflectance(aug_filepath)
        for i in range(10):
            d_S = random.random() * (C.d_S_max - C.d_S_min) + C.d_S_min
            phi = random.randint(C.phi_min, C.phi_max)
            T = random.randint(C.T_min, C.T_max)
            if j % 100 == 0:
                # Calculate radiances with the given parameters and
                # Save plots for every hundredth radiance and reflectance
                observed_radiance(d_S, phi, theta, T, reflectance, waves, 'rads_'+str(j), plots=True)
            else:
                observed_radiance(d_S, phi, theta, T, reflectance, waves, 'rads_' + str(j))

            j = j+1