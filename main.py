
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

    for i in range(0, len(wavelength)): # TODO: nollan voi jättää pois, se on defaultti
        wl = wavelength[i] / 1e6  # Convert wavelength from micrometers to meters
        L_th[i,1] = eps * (2 * h * c**2) / ((wl**5) * (np.exp((h * c)/(wl * kB * T)) - 1))  # Apply Planck's law
        L_th[i,1] = L_th[i,1] / 1e6  # Convert radiance from (W / m² / sr / m) to (W / m² / sr / µm)

    return L_th


def reflected_radiance(reflectance, irradiance, phi): # TODO: muuttujille voi laittaa myös vinkit niiden datatyypista esim. phi: float
    """
    Calculate spectral radiance reflected from a surface, based on the surface reflectance, irradiance incident on it,
    and the phase angle of the measurement. The surface normal is assumed to point towards the observer.

    TODO: jätä rivinvaihto ennenku alotat parametrilistan
    TODO: parametrilistasta tulee siistimpi kun laitat rivinvaihdon ennen selitystä esim.

    :param reflectance: vector of floats.
        Spectral reflectance.

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


def read_meteorites(waves):
    """
    Load reflectance spectra from asteroid analogues measured by Maturilli et al. 2016 (DOI: 10.1186/s40623-016-0489-y)
    and from meteorite spectra measured by Gaffey in 1976 (https://doi.org/10.26033/4nsb-mc72)
    :param waves: a vector of floats, wavelengths to which the data will be interpolated
    :return: a list of Pandas DataArrays containing wavelength vectors and reflectance spectra
    """ # TODO tapana jättää yks rivinvaihto metodikommentin jälkeen ennen ku koodi alkaa
    # Path to folder of reflectance spectra from asteroid analogues
    Maturilli_path = Path('./spectral_data/asteroid_analogues/refle/MIR') # TODO tää polkuluokka on mulle uus. ite oon käyttänu os.path
    MIR_refl_list = os.listdir(Maturilli_path)

    # Path to folder of Gaffey meteorite spectra
    refl_path = Path('./spectral_data/Gaffey_meteorite_spectra/data/spectra')
    Gaffey_refl_list = os.listdir(refl_path)

    reflectances = []  # A table for holding data frames

    for filename in MIR_refl_list:
        filepath = Path.joinpath(Maturilli_path, filename)
        # TODO tässä ois myös siistimpi käyttää sitä regexpii mikä kirjotettiin jonnekin. ihan siltäkin varalta että välilyöntejä ei satu olemaan just tuo määrä
        frame = pd.read_csv(filepath, sep='    ', engine='python') # Read wl and reflectance from file
        frame.columns = ['wavenumber', 'wl', 'reflectance']
        frame.drop('wavenumber', inplace=True, axis=1)  # Drop the wavenumbers, because who uses them anyway

        # frame = frame.loc[frame['wl'] <= 2.5]  # Cut away wl:s above 2.5 µm. Also below 1.5 µm? Noisy.

        # Interpolate reflectance data to match the input wl-vector, and store into new dataFrame
        interp_refl = np.interp(waves, frame.wl.values, frame.reflectance.values)
        data = np.zeros((len(waves),2))
        data[:,0] = waves
        data[:,1] = interp_refl
        frame = pd.DataFrame(data, columns=['wl', 'reflectance'])

        reflectances.append(frame)

        # plt.figure()
        # plt.plot(frame['wl'], frame['reflectance'])
        # plt.show()
        # plt.figure()
        # plt.plot(range(0,len(frame['wl'])), frame['wl'],'*')
        # plt.show()
    # TODO ja tosiaan tee tästä mieluummin apumetodi niin bugien korjaus onnistuu yhtä juttua muuttamalla
    for filename in Gaffey_refl_list:
        if filename.endswith('.tab'):
            filepath = Path.joinpath(refl_path, filename)
            frame = pd.read_table(filepath, sep=' +', header=None, names=('wl', 'reflectance', 'error'), engine='python')
            frame.drop('error', inplace=True, axis=1)  # Drop the error -column, only a few have sensible data there
            frame.wl = frame.wl / 1000  # Convert nm to µm

            # frame = frame.loc[frame['wl'] <= 2.5]  # Only include data from 1 µm to 2.5 µm
            # frame = frame.loc[frame['wl'] >= 1.0]

            # Interpolate reflectance data to match the input wl-vector, and store into new dataFrame
            interp_refl = np.interp(waves, frame.wl.values, frame.reflectance.values)
            data = np.zeros((len(waves), 2))
            data[:, 0] = waves
            data[:, 1] = interp_refl
            frame = pd.DataFrame(data, columns=['wl', 'reflectance'])

            reflectances.append(frame)

            # plt.figure()
            # plt.plot(frame['wl'], frame['R'])
            # plt.show()

        else: continue  # Skip files with extension other than .tab

    return reflectances

def sloper(spectrum):

    return spectrum

def multiplier(spectrum):
    val = np.random.rand(1) + 0.5
    multiplied = spectrum * val
    return multiplied

def offsetter(spectrum):
    val = (np.random.rand(1) - 0.5) * 0.1
    offsetted = spectrum + val
    return offsetted

# def checker_fixer(spectrum):


if __name__ == '__main__':
    # TODO impotit on yleensä tiedoston alussa järjestyksessä 1. python sisäänrakennetut, 2. julkiset moduulit 3. ite tehdyt moduulit
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
    step = 0.002
    waves = np.arange(1, 2.5+step, step=step)  # Create wl-vector from 1 to 2.5 µm, with step given above

    # Load meteorite reflectance spectra from files
    reflectance_spectra = read_meteorites(waves)

    # Augment reflectance spectra with slope, multiplication, and offset
    #TODO First just one for testing, later do this for all
    aug_number = 10  # How many new spectra to generate from each meteorite spectrum
    # augmentation_factors = np.zeros((aug_number, 3)) # np.zeros((len(waves), 3))

    spectrum = reflectance_spectra[0].values

    for i in range(0, aug_number):
        vals = np.random.rand(3)
        s = (vals[0] - 0.5) * 0.1  # Slope
        m = vals[1] + 0.5  # Multiplication
        o = (vals[2] - 0.5) * 0.1  # Offset
        # augmentation_factors[i, :] = [s, m, o]
        spectrum_multiplied = spectrum.copy()
        spectrum_multiplied[:, 1] = spectrum_multiplied[:, 1] * m

        spectrum_multiplied_offset = spectrum_multiplied.copy()
        spectrum_multiplied_offset[:, 1] = spectrum_multiplied_offset[:, 1] + o

        spectrum_sloped = spectrum_multiplied_offset.copy()

        plt.figure()
        plt.plot(spectrum[:,0], spectrum[:,1])
        plt.plot(spectrum_multiplied[:, 0], spectrum_multiplied[:, 1])
        plt.plot(spectrum_multiplied_offset[:, 0], spectrum_multiplied_offset[:, 1])
        plt.show()

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
    interp_insolation = np.zeros(shape=np.shape(reflectance_spectra[0].values))
    interp_insolation[:, 0] = waves
    # TODO tämmöset kovakoodatut stringit tekee koodista vaikeeta muuttaa. tökkää toi wl nimi mieluummin muuttujaan ja käytä sitä
    # TODO mielellään vielä erilliseen filuun joka on sitten importattu kaikkialle missä niitä nimiä tarvitaan. voi tuntuu turhalta
    # TODO vielä ku koodi on nii lyhyt, mutta säästät hermoja myöhemmin
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
    # TODO mainissa aika paljon tauhkaa. siirrä tavara omiksi metodeiksi sitä mukaa kun saat testailtua että ne toimii
    # TODO muuten menee sotkuseks ja aiheuttaa tuskaa myöhemmin






