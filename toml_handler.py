from pathlib import Path
import numpy as np
import toml
import constants as C


def save_aug_reflectance(reflectance: np.ndarray, filename: str):
    """
    Save augmented reflectance to a predetermined folder with the filename given as parameter

    :param reflectance: ndarray
        Reflectance to be saved, with wavelengths in the first column and reflectances in the second

    :param filename: str
        Filename the data will be saved as. The extension (.toml) will be added during saving
    """

    # Create a dict of the reflectance data
    d = {}
    d[C.wl_key] = reflectance[:, 0]
    d[C.R_key] = reflectance[:, 1]

    # Combine given filename with predetermined folder path and save as .toml
    p = C.augmented_path.joinpath(filename + '.toml')
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


def save_radiances(rad_dict, filename):
    rad_path = C.radiance_path
    p = rad_path.joinpath(filename + '.toml')

    with open(p, 'w+') as file:
        toml.dump(rad_dict, file, encoder=toml.encoder.TomlNumpyEncoder())
    print(f'Saved a spectrum into {p}')




