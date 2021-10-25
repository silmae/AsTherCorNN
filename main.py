from os import path
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy import io
import pandas as pd
from solar import solar_irradiance


def bb_radiance(T, eps, wavelength):
    """
    Calculate and return approximate thermal emission (blackbody, bb) radiance spectrum using Planck's law.

    :param T: float. Temperature in Kelvins
    :param eps: float. Emissivity = sample emission spectrum divided by ideal bb spectrum. Maybe in future accepts
    a vector, but only constants for now
    :param wavelength: vector of floats. Wavelengths where the emission is to be calculated, in micrometers.

    :return L_th: vector of floats. Spectral radiance emitted by the surface.
    """

    # Define constants
    c = 2.998e8  # speed of light in vacuum, m / s
    kB = 1.381e-23  # Boltzmann constant, m² kg / s² / K (= J / K)
    h = 6.626e-34  # Planck constant, m² kg / s (= J s)

    L_th = np.zeros((len(wavelength),2))
    L_th[:,0] = wavelength

    for i in range(len(wavelength)):
        wl = wavelength[i] / 1e6  # Convert wavelength from micrometers to meters
        L_th[i,1] = eps * (2 * h * c**2) / ((wl**5) * (np.exp((h * c)/(wl * kB * T)) - 1))  # Apply Planck's law
        L_th[i,1] = L_th[i,1] / 1e6  # Convert radiance from (W / m² / sr / m) to (W / m² / sr / µm)

    return L_th


def reflected_radiance(reflectance: np.ndarray, irradiance: np.ndarray, phi: float):
    """
    Calculate spectral radiance reflected from a surface, based on the surface reflectance, irradiance incident on it,
    and the phase angle of the measurement. The surface normal is assumed to point towards the observer.

    :param reflectance: vector of floats.
        Spectral reflectance.
    :param reflectance: vector of floats.
        Spectral reflectance.
    :param irradiance: vector of floats.
        Spectral irradiance incident on the surface.
    :param phi: float.
        Phase angle of the measurement, in degrees

    :return: vector of floats.
        Spectral radiance reflected toward the observer.
    """
    # TODO Implement changing angle between surface normal and reflected light
    # TODO attach the wavelength vector to the returned data (and the inputs will have it also)
    wavelength = irradiance[:, 0]
    reflrad = np.zeros((len(wavelength), 2))
    reflrad[:, 0] = wavelength

    reflrad[:, 1] = reflectance[:, 1] * (irradiance[:, 1] * np.cos(np.deg2rad(phi))) / np.pi

    return reflrad


def read_meteorites(waves):
    """
    Load reflectance spectra from asteroid analogues measured by Maturilli et al. 2016 (DOI: 10.1186/s40623-016-0489-y)
    and from meteorite spectra measured by Gaffey in 1976 (https://doi.org/10.26033/4nsb-mc72)

    :param waves: a vector of floats
        wavelengths to which the reflectance data will be interpolated

    :return: a list of ndarrays containing wavelength vectors and reflectance spectra
    """

    # Path to folder of reflectance spectra from Maturilli's asteroid analogues
    Maturilli_path = Path('./spectral_data/asteroid_analogues/refle/MIR')
    MIR_refl_list = os.listdir(Maturilli_path)

    # Path to folder of Gaffey meteorite spectra
    refl_path = Path('./spectral_data/Gaffey_meteorite_spectra/data/spectra')
    Gaffey_refl_list = os.listdir(refl_path)

    reflectances = []  # A table for holding data frames

    for filename in MIR_refl_list:
        filepath = Path.joinpath(Maturilli_path, filename)
        frame = pd.read_csv(filepath, sep=' +', header=None, names=('wavenumber', 'wl', 'reflectance'), engine='python') # Read wl and reflectance from file
        frame.columns = ['wavenumber', 'wl', 'reflectance']
        frame.drop('wavenumber', inplace=True, axis=1)  # Drop the wavenumbers, because who uses them anyway

        # Interpolate reflectance data to match the input wl-vector, and store into new dataFrame
        interp_refl = np.interp(waves, frame.wl.values, frame.reflectance.values)
        data = np.zeros((len(waves),2))
        data[:, 0] = waves
        data[:, 1] = interp_refl
        # frame = pd.DataFrame(data, columns=['wl', 'reflectance'])

        # reflectances.append(frame)
        reflectances.append(data)

        # plt.figure()
        # plt.plot(frame['wl'], frame['reflectance'])
        # plt.show()

    # TODO ja tosiaan tee tästä mieluummin apumetodi niin bugien korjaus onnistuu yhtä juttua muuttamalla
    for filename in Gaffey_refl_list:
        if filename.endswith('.tab'):
            filepath = Path.joinpath(refl_path, filename)
            frame = pd.read_table(filepath, sep=' +', header=None, names=('wl', 'reflectance', 'error'), engine='python')
            frame.drop('error', inplace=True, axis=1)  # Drop the error -column, only a few have sensible data there
            frame.wl = frame.wl / 1000  # Convert nm to µm

            # Interpolate reflectance data to match the input wl-vector, and store into new dataFrame
            interp_refl = np.interp(waves, frame.wl.values, frame.reflectance.values)
            data = np.zeros((len(waves), 2))
            data[:, 0] = waves
            data[:, 1] = interp_refl
            # frame = pd.DataFrame(data, columns=['wl', 'reflectance'])

            # reflectances.append(frame)
            reflectances.append(data)

            # plt.figure()
            # plt.plot(frame['wl'], frame['R'])
            # plt.show()

        else: continue  # Skip files with extension other than .tab

    return reflectances


def sloper(spectrum: np.ndarray):
    """
    Takes a spectrum as ndarray, and adds a slope to the reflectance part. The slopiness comes from a random number.

    :param spectrum: ndarray
        Reflectance spectrum, with wl vector in the first column, reflectance in the second

    :return: ndarray
        Sloped spectrum, in the same shape as the one given as parameter
    """
    val = (np.random.rand(1) - 0.5) * 0.1
    slope = np.linspace(-val, val, len(spectrum)).flatten()
    sloped = spectrum.copy()
    sloped[:, 1] = sloped[:, 1] + slope
    return sloped


def multiplier(spectrum: np.ndarray):
    """
    Takes a spectrum as ndarray, and multiplies it by a random number between 0.5 and 1.5

    :param spectrum: ndarray
        Reflectance spectrum, with wl vector in the first column, reflectance in the second
    :return: ndarray
        Multiplied spectrum, in the same shape as the one given as parameter
    """
    val = np.random.rand(1) + 0.5
    multiplied = spectrum.copy()
    multiplied[:, 1] = multiplied[:, 1] * val
    return multiplied


def offsetter(spectrum: np.ndarray):
    """
    Takes a spectrum as ndarray, and offsets it by adding a random number between -0.1 and 0.1 to it

    :param spectrum: ndarray
        Reflectance spectrum, with wl vector in the first column, reflectance in the second
    :return: ndarray
        Offset spectrum, in the same shape as the one given as parameter
    """
    val = (np.random.rand(1) - 0.5) * 0.1
    offsetted = spectrum.copy()
    offsetted[:, 1] = offsetted[:, 1] + val
    return offsetted

def checker_fixer(spectrum: np.ndarray):
    """
    Check a reflectance spectrum for non-physical values: negative of greater than one. Fix negatives by offsetting,
    and then too high values by normalizing.

    :param spectrum: ndarray
        Reflectance spectrum, with wl vector in the first column, reflectance in the second
    :return: ndarray
        Fixed spectrum, in the same shape as the one given as parameter
    """
    # If reflectance has negative values, offset by adding the minimum value to it
    if min(spectrum[:, 1]) < 0:
        spectrum[:, 1] = spectrum[:, 1] + min(spectrum[:, 1])

    # If reflectance has values over 1, normalize by dividing each reflectance with the maximum
    if max(spectrum[:, 1]) > 1:
        spectrum[:, 1] = spectrum[:, 1] / max(spectrum[:, 1])

    return spectrum

if __name__ == '__main__':

    # # Accessing measurements of Ryugu by the Hayabusa2:
    # hdulist = fits.open('hyb2_nirs3_20180710_cal.fit')
    # hdulist.info()
    # ryugu_header = hdulist[0].header
    # print(ryugu_header)
    # ryugu_data = hdulist[0].data
    # # The wavelengths? Find somewhere, is not included in header for some reason
    #
    # mystery_spectra = ryugu_data[1,0:127]
    #
    # plt.figure()
    # plt.plot(mystery_spectra)
    # plt.title('Ryugu')
    # plt.show()

    # Create wl-vector from 1 to 2.5 µm, with step size in µm
    step = 0.002  # µm
    waves = np.arange(1, 2.5+step, step=step)

    # Load meteorite reflectance spectra from files
    reflectance_spectra = read_meteorites(waves)

    # Augment reflectance spectra with slope, multiplication, and offset
    aug_number = 10 # How many new spectra to generate from each meteorite spectrum

    # spectrum = reflectance_spectra[100]
    for j in range(len(reflectance_spectra)):
        spectrum = reflectance_spectra[j]
        for i in range(0, aug_number):

            spectrum_multiplied = multiplier(spectrum)
            spectrum_multiplied_offset = offsetter(spectrum_multiplied)
            spectrum_multiplied_offset_sloped = sloper(spectrum_multiplied_offset)
            spectrum_multiplied_offset_sloped = checker_fixer(spectrum_multiplied_offset_sloped)

            reflectance_spectra.append(spectrum_multiplied_offset_sloped)
            # plt.figure()
            # plt.plot(spectrum[:,0], spectrum[:,1])
            # plt.plot(spectrum_multiplied[:, 0], spectrum_multiplied[:, 1])
            # plt.plot(spectrum_multiplied_offset[:, 0], spectrum_multiplied_offset[:, 1])
            # plt.plot(spectrum_multiplied_offset_sloped[:, 0], spectrum_multiplied_offset_sloped[:, 1])
            # plt.show()

    print('test')

    # Take wavelength vector of one reflectance spectrum, to be used for all the things
    # waves = reflectance_spectra[0]['wl']

    # Calculate insolation (incident solar irradiance) at heliocentric distance of 1 AU
    insolation_1au = solar_irradiance(1)

    # # A constant test reflectance of 0.1
    # reflectance = np.zeros((len(wavelength), 2))
    # reflectance[:, 0] = wavelength
    # reflectance[:, 1] = reflectance[:, 1] + 0.1

    # Interpolating insolation to match the reflectance chosen wavelength vector
    interp_insolation = np.zeros(shape=np.shape(reflectance_spectra[0]))
    interp_insolation[:, 0] = waves
    # TODO tämmöset kovakoodatut stringit tekee koodista vaikeeta muuttaa. tökkää toi wl nimi mieluummin muuttujaan ja käytä sitä
    # TODO mielellään vielä erilliseen filuun joka on sitten importattu kaikkialle missä niitä nimiä tarvitaan. voi tuntuu turhalta
    # TODO vielä ku koodi on nii lyhyt, mutta säästät hermoja myöhemmin
    interp_insolation[:, 1] = np.interp(waves, insolation_1au[:, 0], insolation_1au[:, 1])

    # Take one of the reflectance spectra and use it for calculating theoretical radiance reflected from an asteroid
    spectrum_number = 100  # Which reflectance spectrum to use, from 0 to 170
    reflectance = reflectance_spectra[spectrum_number]
    phase_angle = 30  # degrees
    reflrad = reflected_radiance(reflectance, interp_insolation, phase_angle)

    # Calculate theoretical thermal emission from an asteroid's surface
    T = 400  # Asteroid surface temperature in Kelvins
    eps = 0.9  # Emittance TODO Use Kirchoff's law (eps = 1-R) to get emittance from reflectance? Maybe not.
    thermrad = bb_radiance(T, eps, waves)

    figfolder = Path('./figs')

    plt.figure()
    plt.plot(reflectance[:, 0], reflectance[:, 1])
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Reflectance')
    figpath = Path.joinpath(figfolder , Path('reflectance.png'))
    plt.savefig(figpath)

    plt.figure()
    plt.plot(reflrad[:,0], reflrad[:,1])  # Reflected radiance
    plt.plot(thermrad[:, 0], thermrad[:, 1])  # Thermally emitted radiance
    plt.plot(waves, reflrad[:, 1] + thermrad[:, 1])  # Sum of the two radiances
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Radiance')
    plt.legend(('Reflected', 'Thermal', 'Sum'))
    figpath = Path.joinpath(figfolder, Path('radiances.png'))
    plt.savefig(figpath)
    plt.show()
    # TODO mainissa aika paljon tauhkaa. siirrä tavara omiksi metodeiksi sitä mukaa kun saat testailtua että ne toimii
    # TODO muuten menee sotkuseks ja aiheuttaa tuskaa myöhemmin






