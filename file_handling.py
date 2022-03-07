"""
Methods used for loading files and writing into files
"""

from pathlib import Path
import numpy as np
import toml
import pickle
import csv

import constants as C


def save_pickle(data_dict, path):
    with open(path, 'wb') as file_pi:
        pickle.dump(data_dict, file_pi)
    print(f'Saved a pickle into {path}')


def load_pickle(path):
    with open(path, 'rb') as file_pi:
        data = pickle.load(file_pi)
    return data


def load_csv(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        for row in reader:
            data.append(int(row[0]))  # Take first (and only) element in row, convert to int and append to list
    return data


def save_toml(dictionary: dict, savepath):
    with open(savepath, 'w+') as file:
        toml.dump(dictionary, file, encoder=toml.encoder.TomlNumpyEncoder())
    print(f'Saved a dictionary into {savepath}')


def load_toml(filepath):
    with open(filepath, 'r') as file:
        data = toml.load(file)
    return data


def save_aug_reflectance(reflectance: np.ndarray, filename: str, test: bool):
    """
    Save augmented reflectance to a predetermined folder with the filename given as parameter

    :param reflectance: ndarray
        Reflectance to be saved, with wavelengths in the first column and reflectances in the second

    :param filename: str
        Filename the data will be saved as. The extension (.toml) will be added during saving
    :param test: bool
        is the calculated spectrum for training or testing, affects saving location
    """

    # Create a dict of the reflectance data
    d = {}
    d[C.wl_key] = reflectance[:, 0]
    d[C.R_key] = reflectance[:, 1]

    # Combine given filename with predetermined folder path and save as .toml
    if test == True:
        p = C.augmented_test_path.joinpath(filename + '.toml')
    else:
        p = C.augmented_training_path.joinpath(filename + '.toml')

    with open(p, 'w+') as file:
        toml.dump(d, file, encoder=toml.encoder.TomlNumpyEncoder())
    print(f'Saved a spectrum into {p}')


def read_aug_reflectance(filepath: Path):
    """
    Read a reflectance spectrum from a .toml -file and return it as an ndarray

    :param filepath: Path
        Path to the .toml -file to be read

    :return: ndarray
        Array containing wavelength vector in the first column and reflectance vector in the second
    """

    with open(filepath, 'r') as file:
        reflectance_dict = toml.load(file)

    # Extract wavelength and reflectance vectors from the dict and place into ndarray
    reflectance = np.zeros((len(reflectance_dict[C.wl_key]), 2))
    reflectance[:, 0] = reflectance_dict[C.wl_key]
    reflectance[:, 1] = reflectance_dict[C.R_key]

    return reflectance


def save_radiances(rad_dict: dict, filename: str, test: bool):

    if test == True:
        rad_path = C.radiance_test_path
    else:
        rad_path = C.radiance_training_path

    p = rad_path.joinpath(filename + '.toml')

    with open(p, 'w+') as file:
        toml.dump(rad_dict, file, encoder=toml.encoder.TomlNumpyEncoder())
    print(f'Saved a spectrum into {p}')


def read_radiance(filename: str, test: bool):
    if test == True:
        rad_path = C.radiance_test_path
    else:
        rad_path = C.radiance_training_path

    p = rad_path.joinpath(filename)# + '.toml')

    with open(p, 'r') as file:
        radiance_dict = toml.load(file)

    return radiance_dict


def save_rad_bunch(dict):

    # Combine given filename with predetermined folder path and save as .toml
    p = C.rad_bunch_path
    with open(p, 'w+') as file:
        toml.dump(dict, file, encoder=toml.encoder.TomlNumpyEncoder())
    print(f'Saved radiances into {p}')


def load_rad_bunch():
    p = C.rad_bunch_path
    with open(p, 'r') as file:
        rad_bunch_dict = toml.load(file)

    return rad_bunch_dict






