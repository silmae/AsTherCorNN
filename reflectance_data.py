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


def scale_asteroid_reflectances(normalized_frame: pd.DataFrame, albedos_per_reflectance: int, albedo_frame: pd.DataFrame):
    """
    Scale normalized asteroid reflectance spectra to absolute reflectances using typical albedo values for each
    asteroid class.

    :param normalized_frame: 
        Pandas dataframe of normalized asteroid reflectances, with class of each asteroid included
    :param albedos_per_reflectance:
        How many versions of each reflectance, scaled with different random albedo values
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

        for i in range(albedos_per_reflectance):
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
    test_reflectances = scale_asteroid_reflectances(test_frame, 5, albedo_frame)
    train_reflectances = scale_asteroid_reflectances(train_frame, 5, albedo_frame)

    return train_reflectances, test_reflectances


