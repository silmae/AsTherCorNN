
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
    c = 2.998e8  # speed of light, m / s
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

def analogue_reflectances():
    """
    Load reflectance spectra from asteroid analogues measured by Maturilli et al. 2016 (DOI: 10.1186/s40623-016-0489-y)
    :return: a list of Pandas DataArrays containing wavelength vectors and reflectance spectra
    """
    # Path to folder of reflectance spectra from asteroid analogues
    refl_analogue_path = Path('./spectral_data/asteroid_analogues/refle/MIR')
    MIR_refl_list = os.listdir(refl_analogue_path)

    analogues = []  # A dictionary for holding data frames

    for filename in MIR_refl_list:
        filepath = Path.joinpath(refl_analogue_path, filename)
        frame = pd.read_csv(filepath, sep='    ', engine='python')#.values[:, 1:3]  # Read reflectance from file, leave wavenumber out
        frame.columns = ['wavenumber', 'wl', 'reflectance']
        frame.drop('wavenumber', inplace=True, axis=1)  # Drop the wavenumbers, because who uses them anyway
        frame = frame.loc[frame['wl'] <= 4]  # Cut away wl:s above 4 µm. Also below 1.5 µm? Noisy.

        analogues.append(frame)

        # plt.figure()
        # plt.plot(frame['wl'], frame['reflectance'])
        # plt.show()
        # plt.figure()
        # plt.plot(range(0,len(frame['wl'])), frame['wl'],'*')
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



    reflectance_spectra = analogue_reflectances()
    # print(reflectance_spectra[0]['wl'])
    # Interpolate insolation to match the reflectance data

    irradiance_1au = solar_irradiance(1)
    wavelength = irradiance_1au[:,0]  # Using the irradiance wls is not very nice: will have to interpolate other data to this. Consider something else?

    # plt.figure()
    # plt.plot(range(0, len(irradiance_1au[:,0])), irradiance_1au[:, 0], '*')
    # plt.show()

    reflectance = np.zeros((len(wavelength), 2))
    reflectance[:, 0] = wavelength
    reflectance[:, 1] = reflectance[:, 1] + 0.1  # A constant test reflectance of 0.1

    interp_insolation = np.zeros(shape=np.shape(reflectance_spectra[0].values))
    interp_insolation[:, 0] = reflectance_spectra[0]['wl']

    interp_insolation[:, 1] = np.interp(reflectance_spectra[0]['wl'], wavelength, irradiance_1au[:,1])

    phase_angle = 30  # degrees

    reflectance = reflectance_spectra[5].values
    reflrad = reflected_radiance(reflectance, interp_insolation, phase_angle)

    T = 400  # Asteroid surface temperature in Kelvins
    eps = 0.9  # Emittance
    L_th = bb_radiance(T, eps, wavelength)

    plt.figure()
    plt.plot(L_th[:,0], L_th[:,1])

    plt.figure()
    plt.plot(reflectance[:,0], reflectance[:,1])

    plt.figure()
    plt.plot(reflrad[:,0], reflrad[:,1])
    plt.plot(L_th[:, 0], L_th[:, 1])
    plt.show()






