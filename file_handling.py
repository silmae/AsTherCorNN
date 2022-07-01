"""
Methods used for loading files and writing into files
"""

from pathlib import Path
import numpy as np
import toml
import pickle
import csv

import constants as C


def save_pickle(data_dict: dict, path):
    """Save dictionary as a pickle file"""

    with open(path, 'wb') as file_pi:
        pickle.dump(data_dict, file_pi)
    print(f'Saved a pickle into {path}')


def load_pickle(path):
    """Load dictionary from pickle file and return it"""

    with open(path, 'rb') as file_pi:
        data = pickle.load(file_pi)
    return data


def load_csv(filepath):
    """Load csv from file, return a list first element in every row. Used in loading Bennu discard indices. """

    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        for row in reader:
            data.append(int(row[0]))  # Take first (and only) element in row, convert to int and append to list
    return data


def save_toml(dictionary: dict, savepath):
    """Save a dictionary into a .toml file"""

    # Add file extension if not present
    if not(savepath.endswith('.toml')):
        savepath = savepath + '.toml'

    with open(savepath, 'w+') as file:
        toml.dump(dictionary, file, encoder=toml.encoder.TomlNumpyEncoder())
    print(f'Saved a dictionary into {savepath}')


def load_toml(filepath):
    """Load a dictionary from a .toml file"""

    # Add file extension if not present
    if not(filepath.endswith('.toml')):
        filepath = filepath + '.toml'

    with open(filepath, 'r') as file:
        data = toml.load(file)
    return data


def save_radiances(rad_dict: dict, filename: str, test: bool):
    """Saves a dictionary of radiances on disc. Save path determined by filename and parameter test: if test is true,
    saved into a folder of test radiances. If false, into folder of training radiances. """

    if test == True:
        rad_path = C.radiance_test_path
    else:
        rad_path = C.radiance_training_path

    p = rad_path.joinpath(filename + '.toml')

    save_toml(rad_dict, p)


def read_radiance(filename: str, test: bool):
    """Reads a radiance dict from disc. Path of read file determined by filename, and test -parameter. If test is true,
    read from a folder of test radiances. If false, from a folder of training radiances."""

    if test == True:
        rad_path = C.radiance_test_path
    else:
        rad_path = C.radiance_training_path

    p = rad_path.joinpath(filename + '.toml')

    radiance_dict = load_toml(p)

    return radiance_dict
