
def bb_radiance(T, eps, wavelength):
    """
    Calculate and return approximate thermal emission (blackbody, bb) radiance spectrum using Planck's law.
    :param T: float. Temperature in Kelvins
    :param eps: float. Emissivity = sample emission spectrum divided by ideal bb spectrum. Maybe in future accepts
    a vector, but only constants for now
    :param wavelength: vector of floats. Wavelengths where the emission is to be calculated, in nanometers.
    :return L_th: vector of floats. Spectral radiance emitted by the surface.
    """
    # TODO attach the wavelength vector to the returned data
    # Define constants
    c = 2.998e8  # speed of light, m / s
    kB = 1.381e-23  # Boltzmann constant, m² kg / s² / K (= J / K)
    h = 6.626e-34  # Planck constant, m² kg / s (= J s)

    L_th = np.zeros(np.shape(wavelength))

    for i in range(0, len(wavelength)):
        wl = wavelength[i] / 1e9  # Convert wavelength from nanometers to meters
        L_th[i] = eps * (2 * h * c**2) / ((wl**5) * (np.exp((h * c)/(wl * kB * T)) - 1))  # Apply Planck's law
        L_th[i] = L_th[i] / 1e9  # Convert radiance from (W / m² / sr / m) to (W / m² / sr / nm)

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
    refrad = reflectance * (irradiance * np.cosd(phi)) / np.pi

    return refrad

if __name__ == '__main__':
    import numpy as np
    from os import path
    from matplotlib import pyplot as plt
    from solar import solar_irradiance

    irradiance_1au = solar_irradiance(1)

    T = 300
    eps = 0.9
    L_th = bb_radiance(T, eps, irradiance_1au[:,0])

    plt.figure()
    plt.plot(irradiance_1au[:,0], L_th)
    plt.show()






