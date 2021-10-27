import numpy as np
import matplotlib.pyplot as plt
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

    mu = C.mu  # mean and standard deviation
    sigma = C.sigma

    s = np.random.default_rng().normal(mu, sigma, len(rad_data))

    rad_data[:, 1] = rad_data[:, 1] + s

    return rad_data


def observed_radiance(d_S: float, phi: float, theta: float, T: float, reflectance: np.ndarray, waves: np.ndarray, filename: str):

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
    meta = {}
    meta['heliocentric_distance'] = d_S
    meta['phase_angle'] = phi
    meta['emission_angle'] = theta
    meta['surface_temperature'] = T
    meta['emittance'] = eps
    rad_dict['metadata'] = meta
    rad_dict['wavelength'] = waves
    rad_dict['reflected_radiance'] = reflrad[:, 1]
    rad_dict['emitted_radiance'] = thermrad[:, 1]
    rad_dict['sum_radiance'] = sumrad[:, 1]

    # Save the dict as .toml
    tomler.save_radiances(rad_dict, filename)

    # Plotting reflectance and radiances, saving as .png
    figfolder = C.figfolder

    plt.figure()
    plt.plot(reflectance[:, 0], reflectance[:, 1])
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Reflectance')
    figpath = figfolder.joinpath(filename + '_reflectance.png')
    plt.savefig(figpath)

    plt.figure()
    plt.plot(reflrad[:,0], reflrad[:,1])  # Reflected radiance
    plt.plot(thermrad[:, 0], thermrad[:, 1])  # Thermally emitted radiance
    plt.plot(waves, reflrad[:, 1] + thermrad[:, 1])  # Sum of the two radiances
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Radiance')
    plt.legend(('Reflected', 'Thermal', 'Sum'))
    figpath = figfolder.joinpath(filename + ('_radiances.png'))
    plt.savefig(figpath)
    # plt.show()

