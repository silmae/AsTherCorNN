import os
from pathlib import Path
import numpy as np
import pandas as pd
import constants as C
import toml_handler as tomler


def read_meteorites(waves):
    """
    Load reflectance spectra from asteroid analogues measured by Maturilli et al. 2016 (DOI: 10.1186/s40623-016-0489-y)
    and from meteorite spectra measured by Gaffey in 1976 (https://doi.org/10.26033/4nsb-mc72)

    :param waves: a vector of floats
        wavelengths to which the reflectance data will be interpolated

    :return: a list of ndarrays containing wavelength vectors and reflectance spectra
    """

    # Path to folder of reflectance spectra from Maturilli's asteroid analogues
    Maturilli_path = C.Maturilli_path
    MIR_refl_list = os.listdir(Maturilli_path)

    # Path to folder of Gaffey meteorite spectra
    refl_path = C.Gaffey_path
    Gaffey_refl_list = os.listdir(refl_path)

    reflectances = []  # A table for holding data frames

    for filename in MIR_refl_list:
        filepath = Path.joinpath(Maturilli_path, filename)
        frame = pd.read_csv(filepath, sep=' +', header=None, names=('wavenumber', C.wl_key, C.R_key), engine='python') # Read wl and reflectance from file
        # frame.columns = ['wavenumber', C.wl_key, C.R_key]
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

    # TODO tee tästä mieluummin apumetodi niin bugien korjaus onnistuu yhtä juttua muuttamalla
    for filename in Gaffey_refl_list:
        if filename.endswith('.tab'):
            filepath = Path.joinpath(refl_path, filename)
            frame = pd.read_table(filepath, sep=' +', header=None, names=(C.wl_key, C.R_key, 'error'), engine='python')
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


def augmented_reflectances(waves: np.ndarray):
    """
    Load meteorite reflectance spectra from Maturilli's and Gaffey's data, and create more spectra through augmentation.
    Save the created spectra into separate .toml -files.

    :param waves:
        wavelength vector to which the loaded reflectance spectra will be interpolated
    """
    # Load meteorite reflectance spectra from files, interpolating to match the wavelength vector
    reflectance_spectra = read_meteorites(waves)

    # Augment reflectance spectra with slope, multiplication, and offset
    aug_number = 10  # How many new spectra to generate from each meteorite spectrum

    for j in range(len(reflectance_spectra)):
        spectrum = reflectance_spectra[j]
        for i in range(aug_number):
            spectrum_multiplied = multiplier(spectrum)
            spectrum_multiplied_offset = offsetter(spectrum_multiplied)
            spectrum_multiplied_offset_sloped = sloper(spectrum_multiplied_offset)
            spectrum_multiplied_offset_sloped = checker_fixer(spectrum_multiplied_offset_sloped)

            # Add the spectrum to the list
            reflectance_spectra.append(spectrum_multiplied_offset_sloped)

            # plt.figure()
            # plt.plot(spectrum[:,0], spectrum[:,1])
            # plt.plot(spectrum_multiplied[:, 0], spectrum_multiplied[:, 1])
            # plt.plot(spectrum_multiplied_offset[:, 0], spectrum_multiplied_offset[:, 1])
            # plt.plot(spectrum_multiplied_offset_sloped[:, 0], spectrum_multiplied_offset_sloped[:, 1])
            # plt.show()

    # Save the augmented spectra into toml -files
    for j in range(len(reflectance_spectra)):
        spectrum = reflectance_spectra[j]
        tomler.save_aug_reflectance(spectrum, f'reflectance{j}')



