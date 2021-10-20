
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

    for i in range(0, len(wavelength)):
        wl = wavelength[i] / 1e6  # Convert wavelength from micrometers to meters
        L_th[i,1] = eps * (2 * h * c**2) / ((wl**5) * (np.exp((h * c)/(wl * kB * T)) - 1))  # Apply Planck's law
        L_th[i,1] = L_th[i,1] / 1e6  # Convert radiance from (W / m² / sr / m) to (W / m² / sr / µm)

    return L_th


def reflected_radiance(reflectance, irradiance, phi):
    """
    Calculate spectral radiance reflected from a surface, based on the surface reflectance, irradiance incident on it,
    and the phase angle of the measurement. The surface normal is assumed to point towards the observer.
    :param reflectance: vector of floats. Spectral reflectance.
    :param irradiance: vector of floats. Spectral irradiance incident on the surface.
    :param phi: float. Phase angle of the measurement, in degrees
    :return: vector of floats. Spectral radiance reflected toward the observer.
    """
    # TODO Implement changing angle between surface normal and reflected light
    # TODO attach the wavelength vector to the returned data (and the inputs will have it also)
    wavelength = irradiance[:, 0]
    reflrad = np.zeros((len(wavelength), 2))
    reflrad[:, 0] = wavelength

    reflrad[:, 1] = reflectance[:, 1] * (irradiance[:, 1] * np.cos(np.deg2rad(phi))) / np.pi

    return reflrad

def read_Maturilli():
    """
    Load reflectance spectra from asteroid analogues measured by Maturilli et al. 2016 (DOI: 10.1186/s40623-016-0489-y)
    :return: a list of Pandas DataArrays containing wavelength vectors and reflectance spectra
    """
    # Path to folder of reflectance spectra from asteroid analogues
    refl_analogue_path = Path('./spectral_data/asteroid_analogues/refle/MIR')
    MIR_refl_list = os.listdir(refl_analogue_path)

    analogues = []  # A table for holding data frames

    for filename in MIR_refl_list:
        filepath = Path.joinpath(refl_analogue_path, filename)
        frame = pd.read_csv(filepath, sep='    ', engine='python')#.values[:, 1:3]  # Read reflectance from file, leave wavenumber out
        frame.columns = ['wavenumber', 'wl', 'reflectance']
        frame.drop('wavenumber', inplace=True, axis=1)  # Drop the wavenumbers, because who uses them anyway
        frame = frame.loc[frame['wl'] <= 2.5]  # Cut away wl:s above 2.5 µm. Also below 1.5 µm? Noisy.

        analogues.append(frame)

        # plt.figure()
        # plt.plot(frame['wl'], frame['reflectance'])
        # plt.show()
        # plt.figure()
        # plt.plot(range(0,len(frame['wl'])), frame['wl'],'*')
        # plt.show()

    return analogues

def read_Gaffey():

    # Path to folder of meteorite reflectance spectra
    refl_path = Path('./spectral_data/Gaffey_meteorite_spectra/data/spectra')
    Gaffey_refl_list = os.listdir(refl_path)
    analogues = []  # A table for holding data frames

    for filename in Gaffey_refl_list:
        if filename.endswith('.tab'):
            filepath = Path.joinpath(refl_path, filename)
            data = pd.read_table(filepath, sep=' +', header=None, names=('wl', 'R', 'error'), engine='python')
            data.drop('error', inplace=True, axis=1)
            analogues.append(data)

            plt.figure()
            plt.plot(data['wl'], data['R'])
            plt.show()

        else: continue

    # plt.figure()
    # plt.plot(range(0,len(data['wl'])), data['wl'],'*')
    # plt.show()

    return analogues

if __name__ == '__main__':
    import numpy as np
    from os import path
    import os
    from matplotlib import pyplot as plt
    from solar import solar_irradiance
    from astropy.io import fits
    from pathlib import Path
    from scipy import io
    import pandas as pd

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

    waves = np.arange(1, 2.505, step = 0.002)  # Create wl-vector from 1 to 2.5 µm, with 5 nm step

    Gaffey_spectra = read_Gaffey()

    # Load seven meteorite reflectance spectra from files
    reflectance_spectra = read_Maturilli()

    # Take wavelength vector of one reflectance spectrum, to be used for all the things
    # waves = reflectance_spectra[0]['wl']

    # Calculate insolation (incident solar irradiance) at heliocentric distance of 1 AU
    insolation_1au = solar_irradiance(1)

    # # A constant test reflectance of 0.1
    # reflectance = np.zeros((len(wavelength), 2))
    # reflectance[:, 0] = wavelength
    # reflectance[:, 1] = reflectance[:, 1] + 0.1

    # Interpolating insolation to match the reflectance chosen wavelength vector
    interp_insolation = np.zeros(shape=np.shape(reflectance_spectra[0].values))
    interp_insolation[:, 0] = waves

    interp_insolation[:, 1] = np.interp(reflectance_spectra[0]['wl'], insolation_1au[:, 0], insolation_1au[:, 1])

    # Take one of the reflectance spectra and use it for calculating theoretical radiance reflected from an asteroid
    spectrum_number = 2  # Which reflectance spectrum to use, from 0 to 6
    reflectance = reflectance_spectra[spectrum_number].values
    phase_angle = 30  # degrees
    reflrad = reflected_radiance(reflectance, interp_insolation, phase_angle)

    # Calculate theoretical thermal emission from an asteroid's surface
    T = 400  # Asteroid surface temperature in Kelvins
    eps = 0.9  # Emittance TODO Use Kirchoff's law (eps = 1-R) to get emittance from reflectance? Maybe not.
    thermrad = bb_radiance(T, eps, waves)

    figfolder = Path('./figs')

    plt.figure()
    plt.plot(reflectance[:,0], reflectance[:,1])
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






