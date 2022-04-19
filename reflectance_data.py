"""
Methods for working with reflectance data: loading and augmenting meteorite reflectances, loading asteroid reflectances
and scaling them with appropriate albedo estimates.
"""

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import constants as C
import file_handling as FH


def read_meteorites(waves):
    """
    Load reflectance spectra from asteroid analogues measured by Maturilli et al. 2016 (DOI: 10.1186/s40623-016-0489-y)
    and from meteorite spectra measured by Gaffey in 1976 (https://doi.org/10.26033/4nsb-mc72)
    Shuffle the order of reflectances, and partition to training and testing
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

    reflectances = []  # A list for holding data frames

    for filename in MIR_refl_list:
        filepath = Path.joinpath(Maturilli_path, filename)
        frame = pd.read_csv(filepath, sep=' +', header=None, names=('wavenumber', C.wl_key, C.R_key), engine='python') # Read wl and reflectance from file
        # frame.columns = ['wavenumber', C.wl_key, C.R_key]
        frame.drop('wavenumber', inplace=True, axis=1)  # Drop the wavenumbers, because who uses them anyway

        # Interpolate reflectance data to match the input wl-vector, and store into new dataFrame
        interp_refl = np.interp(waves, frame[C.wl_key].values, frame[C.R_key].values)
        data = np.zeros((len(waves),2))
        data[:, 0] = waves
        data[:, 1] = interp_refl
        # frame = pd.DataFrame(data, columns=['wl', 'reflectance'])

        # reflectances.append(frame)
        reflectances.append(data)

        # plt.figure()
        # plt.plot(data[:, 0], data[:, 1])
        # plt.show()

    # TODO tee tästä mieluummin apumetodi niin bugien korjaus onnistuu yhtä juttua muuttamalla
    for filename in Gaffey_refl_list:
        if filename.endswith('.tab'):
            filepath = Path.joinpath(refl_path, filename)
            frame = pd.read_table(filepath, sep=' +', header=None, names=(C.wl_key, C.R_key, 'error'), engine='python')
            frame.drop('error', inplace=True, axis=1)  # Drop the error -column, only a few have sensible data there
            frame[C.wl_key] = frame[C.wl_key] / 1000  # Convert nm to µm

            # Interpolate reflectance data to match the input wl-vector, and store into new dataFrame
            interp_refl = np.interp(waves, frame[C.wl_key].values, frame[C.R_key].values)
            data = np.zeros((len(waves), 2))
            data[:, 0] = waves
            data[:, 1] = interp_refl
            # frame = pd.DataFrame(data, columns=['wl', 'reflectance'])

            # reflectances.append(frame)
            reflectances.append(data)

            # plt.figure()
            # plt.plot(data[:, 0], data[:, 1])
            # plt.xlabel('Wavelength [µm]')
            # plt.ylabel('Reflectance')
            # plt.savefig(Path.joinpath(C.figfolder, filename+'.png'))
            # # plt.show()

        else: continue  # Skip files with extension other than .tab

    # Shuffle the order of the reflectance list
    random.shuffle(reflectances)

    # Take 30% for testing and 70% for training
    sample_count = len(reflectances)
    test_part = int(sample_count * 0.3)
    test_reflectances = reflectances[:test_part]
    train_reflectances = reflectances[test_part:]

    return train_reflectances, test_reflectances


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

    # If reflectance has negative values, offset by adding the absolute value of the minimum to each element
    if min(spectrum[:, 1]) < 0:
        spectrum[:, 1] = spectrum[:, 1] + abs(min(spectrum[:, 1]))

    # If reflectance has values over 1, normalize by dividing each reflectance with the maximum
    if max(spectrum[:, 1]) > 1:
        spectrum[:, 1] = spectrum[:, 1] / max(spectrum[:, 1])

    return spectrum


def augmented_reflectances(reflectance_spectra: list, waves: np.ndarray, test: bool):
    """
    Load meteorite reflectance spectra from Maturilli's and Gaffey's data, and create more spectra through augmentation.
    Save the created spectra into separate .toml -files.

    :param waves:
        wavelength vector to which the loaded reflectance spectra will be interpolated
    :param reflectance_spectra:
        list of reflectance spectra to be augmented
    :param test:
        is the spectrum from testing or training pool, affects save location
    """

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
        FH.save_aug_reflectance(spectrum, f'reflectance{j}', test)


def scale_asteroid_reflectances(normalized_frame: pd.DataFrame, albedo_frame: pd.DataFrame):
    """
    Scale normalized asteroid reflectance spectra to absolute reflectances using typical albedo values for each
    asteroid class.

    :param normalized_frame: 
        Pandas dataframe of normalized asteroid reflectances, with class of each asteroid included
    :param albedo_frame:
        Pandas dataframe of typical albedos for asteroid classes, including a range of variation for each
    
    :return: list
        Scaled spectra reflectances in a list
    """

    # Empty list for the reflectances
    spectral_reflectances = []

    plot_index = 0
    # Convert dataframe to ndarray and iterate over the rows
    for row in normalized_frame.values:
        # The first value of a row is the asteroid class, the rest is normalized reflectance
        asteroid_class, norm_reflectance = row[0], row[1:]

        # Geometrical albedo, pulled from uniform distribution between min and max
        geom_albedo = random.uniform(C.p_min, C.p_max)

        # Un-normalize reflectance by scaling it with visual geometrical albedo
        spectral_reflectance = norm_reflectance * geom_albedo
        # Convert reflectance to single-scattering albedo, using Lommel-Seeliger
        spectral_single_scattering_albedo = 8 * spectral_reflectance

        # Print if the physical limits of min and max reflectance are exceeded. And they will be, since L-S is not
        # really suitable for bodies with geom. albedo > 0.125
        if np.max(spectral_single_scattering_albedo) > 1 or np.min(spectral_single_scattering_albedo) < 0:
            print(f'Unphysical reflectance detected! Max {np.max(spectral_single_scattering_albedo)}, min {np.min(spectral_single_scattering_albedo)}')

        spectral_reflectances.append(spectral_single_scattering_albedo)

        # Plot every hundreth set of three reflectances, save plots to disc
        if plot_index % 100 == 0:
            fig = plt.figure()
            plt.plot(C.wavelengths, spectral_reflectances[-1])
            plt.plot(C.wavelengths, spectral_reflectances[-2])
            plt.plot(C.wavelengths, spectral_reflectances[-3])
            plt.xlabel('Wavelength [µm]')
            plt.ylabel('Reflectance')
            plt.savefig(Path(C.refl_plots_path, f'{plot_index}.png'), dpi=400)
            plt.close(fig)

        plot_index = plot_index + 1

    return spectral_reflectances


def read_asteroids():
    """
    Read asteroid reflectances from a file, and normalize them using typical albedos of each asteroid type. The
    asteroid reflectance data was provided by A. Penttilä, and the dataset is described in
    DOI: 10.1051/0004-6361/202038545

    :return: training reflectance list, test reflectance list
    Scaled reflectance spectra, partitioned according to split given in constants.py

    """
    aug_path = C.Penttila_aug_path  # Spectra augmented by Penttilä
    orig_path = C.Penttila_orig_path  # Un-augmented, original spectra from MITHNEOS and Bus-Demeo

    aug_frame = pd.read_csv(aug_path, sep='\t', header=None, engine='python')  # Read wl and reflectance from file
    orig_frame = pd.read_csv(orig_path, sep='\t', header=None, engine='python')
    albedo_frame = pd.read_csv(C.albedo_path, sep='\t', header=None, engine='python', index_col=0)  # Read mean albedos for classes

    # Extract wl vector from the original: the same as in augmented, but that one does not have it
    wavelengths = orig_frame.values[0, 2:]

    # Shuffling the order of rows, so all spectra of the same type won't occur one after another
    aug_frame = aug_frame.sample(frac=1)

    # Partition the data into train and test reflectances, according to split parameter given in constants
    sample_count = len(aug_frame.index)
    test_part = int(sample_count * C.refl_test_partition)

    test_frame = aug_frame.iloc[:test_part, :]
    train_frame = aug_frame.iloc[test_part:, :]

    # Scale normalized spectra using class mean albedos
    test_reflectances = scale_asteroid_reflectances(test_frame, albedo_frame)
    train_reflectances = scale_asteroid_reflectances(train_frame, albedo_frame)

    return train_reflectances, test_reflectances


