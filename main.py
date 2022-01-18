import random
from os import path
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'savefig.dpi': 600})
# from astropy.io import fits
from scipy import io
import pandas as pd
from tensorflow import keras
import pickle
from sklearn.model_selection import train_test_split

from solar import solar_irradiance
import constants as C
import reflectance_data as refl
import radiance_data as rad
import toml_handler as tomler
import neural_network as NN
import validation as val


if __name__ == '__main__':

    # #######################
    # # For running with GPU on server:
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # # Check available GPU with command nvidia-smi in terminal, pick one that is not in use
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    #
    # # After you have started your computing task please use "nvidia-smi" command
    # # and check that your program has correctly reserved GPU memory and that it
    # # actually runs in GPU(s).
    # #
    # # Memory usage is in the middle column and GPU usage is in the rightmost co-
    # # lumn. If GPU usage shows 0% then your code runs only in CPU, not in GPU.
    #
    # # To use plt.show() from server, make X11 connection and add this to Run configuration > Environment variables:
    # # DISPLAY=localhost:10.0
    # ############################

    # # Opening OVIRS spectra measured from Bennu
    # Bennu_path = Path(C.spectral_path, 'Bennu_OVIRS')
    # file_list = os.listdir(Bennu_path)
    #
    # # Group files by time of day: 20190425 is 3pm data, 20190509 is 12 pm data, 20190516 is 10 am data
    # # Empty lists with two elements each for holding the fits data
    # Bennu_10 = [None] * 2
    # Bennu_12 = [None] * 2
    # Bennu_15 = [None] * 2
    # for filename in file_list:
    #     filepath = Path(Bennu_path, filename)
    #     # Open .fits file with astropy, append to a list
    #
    #     # A is uncorrected radiance, B is thermal tail removed radiance: make first element the uncorrected
    #     if 'A' in filename:
    #         index = 0
    #     elif 'B' in filename:
    #         index = 1
    #
    #     if '20190425' in filename:
    #         hdulist = fits.open(filepath)
    #         Bennu_15[index] = hdulist
    #     elif '20190509' in filename:
    #         hdulist = fits.open(filepath)
    #         Bennu_12[index] = hdulist
    #     elif '20190516' in filename:
    #         hdulist = fits.open(filepath)
    #         Bennu_10[index] = hdulist
    #
    # def bennu_refine(fitslist):
    #
    #     uncorrected_fits = fitslist[0]
    #     corrected_fits = fitslist[1]
    #
    #     # # Handy info print of what the fits file includes:
    #     # corrected_fits.info()
    #
    #     wavelengths = uncorrected_fits[1].data
    #     # header = corrected_fits[0].header
    #     uncorrected_rad = uncorrected_fits[0].data[:, 0, :]
    #     corrected_rad = corrected_fits[0].data[:, 0, :]
    #
    #     uncor_sum_rad = np.sum(uncorrected_rad, 1)
    #     cor_sum_rad = np.sum(corrected_rad, 1)
    #
    #     # Data is from several scans over Bennu's surface, each scan beginning and ending off-asteroid. See plot of
    #     # radiances summed over wl:s:
    #     plt.figure()
    #     plt.plot(range(len(uncor_sum_rad)), uncor_sum_rad)
    #     plt.xlabel('Measurement number')
    #     plt.ylabel('Radiance summed over wavelengths')
    #     plt.show()
    #
    #     # Go over the summed uncorrected radiances, and save the indices where radiance is over 0.02 (value from plots):
    #     # gives indices of datapoints where the FOV was on Bennu
    #     Bennu_indices = []
    #     index = 0
    #     for sum_rad in uncor_sum_rad:
    #         if sum_rad > 0.02:
    #             Bennu_indices.append(index)
    #         index = index + 1
    #
    #     # Pick out the spectra where sum radiance was over threshold value
    #     uncorrected_Bennu = uncorrected_rad[Bennu_indices, :]
    #     corrected_Bennu = corrected_rad[Bennu_indices, :]
    #
    #     plt.figure()
    #     plt.plot(wavelengths, uncorrected_Bennu[0, :])
    #     plt.plot(wavelengths, corrected_Bennu[0, :])
    #
    #     plt.show()
    #
    #     foo = 0
    #
    # bennu_refine(Bennu_15)
    # bennu_refine(Bennu_12)
    # bennu_refine(Bennu_10)

    #############################

    # # # Plot uncorrected and (ideally) corrected reflectance from one radiance sample to illustrate why this is relevant
    # # rad_dict = tomler.read_radiance('rads_5700.toml')
    # # meta = rad_dict['metadata']
    # # uncorrected = rad.radiance2reflectance(rad_dict['sum_radiance'], meta['heliocentric_distance'], meta['phase_angle'], meta['emission_angle'])
    # # corrected = rad.radiance2reflectance(rad_dict['reflected_radiance'], meta['heliocentric_distance'], meta['phase_angle'], meta['emission_angle'])
    # #
    # # plt.figure()
    # # plt.plot(C.wavelengths, corrected)
    # # plt.plot(C.wavelengths, uncorrected)
    # # plt.legend(('Corrected', 'Uncorrected'))
    # # plt.xlabel('Wavelength [µm]')
    # # plt.ylabel('Reflectance')
    # # plt.show()
    # # ##############################

    # # Build and train a model
    # model = NN.train_autoencoder(early_stop=True, checkpoints=True, save_history=True)
    # model = NN.train_autoencoder(early_stop=False, checkpoints=True, save_history=True)

    ##############################

    # Build a model and load pre-trained weights
    model = NN.load_model(Path(C.weights_path, 'weights_992.hdf5'))
    # model.summary()

    # val.validate_synthetic(model)
    val.validate_bennu(model)






