import random
from os import path
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy import io
import pandas as pd
from tensorflow import keras

from solar import solar_irradiance
import constants as C
import reflectance_data as refl
import radiance_data as rad
import toml_handler as tomler
import neural_network as NN


if __name__ == '__main__':

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

    # Create wl-vector from 1 to 2.5 µm, with step size in µm
    waves = C.wavelengths

    ##############################
    # Load meteorite reflectances from files and create more from them through augmentation
    # refl.augmented_reflectances(waves)
    # foo = refl.read_meteorites(waves)
    ##############################

    # rad.calculate_radiances()












