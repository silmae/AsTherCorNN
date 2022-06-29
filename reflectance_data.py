"""
Methods for working with reflectance data: loading and augmenting meteorite reflectances, loading asteroid reflectances
and scaling them with appropriate albedo estimates.
"""

import random
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import constants as C


def read_asteroids():
    """
    Read asteroid reflectances from a file, and normalize them using random albedo values. The
    asteroid reflectance data was provided by A. Penttilä, and the dataset is described in
    DOI: 10.1051/0004-6361/202038545

    :return: training reflectance list, test reflectance list
    Scaled reflectance spectra, partitioned according to split given in constants.py

    """

    aug_path = C.Penttila_aug_path  # Spectra augmented by Penttilä

    aug_frame = pd.read_csv(aug_path, sep='\t', header=None, engine='python')  # Read reflectance from file

    # # Extract wl vector from the original: the same as in augmented, but that one does not have it
    # orig_path = C.Penttila_orig_path  # Un-augmented, original spectra from MITHNEOS and Bus-Demeo
    # orig_frame = pd.read_csv(orig_path, sep='\t', header=None, engine='python')
    # wavelengths = orig_frame.values[0, 2:]

    # Shuffling the order of rows, so all spectra of the same type won't occur one after another
    aug_frame = aug_frame.sample(frac=1)

    # Partition the data into train and test reflectances, according to split parameter given in constants
    sample_count = len(aug_frame.index)
    test_part = int(sample_count * C.refl_test_partition)

    # Dividing data into test and training sets
    test_frame = aug_frame.iloc[:test_part, :]
    train_frame = aug_frame.iloc[test_part:, :]

    # Scale normalized spectra using random albedos
    test_reflectances = _scale_asteroid_reflectances(test_frame, 5)
    train_reflectances = _scale_asteroid_reflectances(train_frame, 5)

    return train_reflectances, test_reflectances


def _scale_asteroid_reflectances(normalized_frame: pd.DataFrame, albedos_per_reflectance: int):
    """
    Scale normalized asteroid reflectance spectra to absolute reflectances using typical albedo values for each
    asteroid class.

    :param normalized_frame:
        Pandas dataframe of normalized asteroid reflectances, with class of each asteroid included
    :param albedos_per_reflectance:
        How many versions of each reflectance, scaled with different random albedo values

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

        for i in range(albedos_per_reflectance):
            # Geometrical albedo, pulled from uniform distribution between min and max
            geom_albedo = random.uniform(C.p_min, C.p_max)

            # Un-normalize reflectance by scaling it with visual geometrical albedo
            spectral_reflectance = norm_reflectance * geom_albedo
            # Convert reflectance to single-scattering albedo, using Lommel-Seeliger
            spectral_single_scattering_albedo = 8 * spectral_reflectance

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
