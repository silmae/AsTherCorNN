
def bb_radiance(T, eps, wavelength):
    """
    Calculate and return approximate thermal emission (blackbody, bb) radiance spectrum using Planck's law.
    :param T: float. Temperature in Kelvins
    :param eps: float. Emissivity = sample emission spectrum divided by ideal bb spectrum. Maybe in future accepts
    a vector, but only constants for now
    :param wavelength: vector of floats. Wavelengths where the emission is to be calculated, in nanometers.
    :return L_th: vector of floats. Spectral radiance emitted by the surface.
    """

    # Define constants
    c = 2.998e8  # speed of light, m / s
    kB = 1.381e-23  # Boltzmann constant, m² kg / s² / K (= J / K)
    h = 6.626e-34  # Planck constant, m² kg / s (= J s)

    L_th = np.zeros((len(wavelength),2))
    L_th[:,0] = wavelength

    for i in range(0, len(wavelength)):
        wl = wavelength[i] / 1e9  # Convert wavelength from nanometers to meters
        L_th[i,1] = eps * (2 * h * c**2) / ((wl**5) * (np.exp((h * c)/(wl * kB * T)) - 1))  # Apply Planck's law
        L_th[i,1] = L_th[i,1] / 1e9  # Convert radiance from (W / m² / sr / m) to (W / m² / sr / nm)

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

if __name__ == '__main__':
    import numpy as np
    from os import path
    from matplotlib import pyplot as plt
    from solar import solar_irradiance
    from astropy.io import fits

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

    irradiance_1au = solar_irradiance(1)
    wavelength = irradiance_1au[:,0]  # Using the irradiance wls is not very nice: will have to interpolate other data to this. Consider something else.

    reflectance = np.zeros((len(wavelength), 2))
    reflectance[:, 0] = wavelength
    reflectance[:, 1] = reflectance[:, 1] + 0.1
    # reflectance[:, 1] = np.log(wavelength)  # A test reflectance
    # reflectance[:, 1] = reflectance[:, 1] / max(reflectance[:, 1])

    phase_angle = 30  # degrees

    reflrad = reflected_radiance(reflectance, irradiance_1au, phase_angle)

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






