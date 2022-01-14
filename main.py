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
import pickle
from sklearn.model_selection import train_test_split

from solar import solar_irradiance
import constants as C
import reflectance_data as refl
import radiance_data as rad
import toml_handler as tomler
import neural_network as NN

# # For running with GPU on server:
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # Check available GPU with command nvidia-smi in terminal
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#######################
# After you have started your computing task please use "nvidia-smi" command
# and check that your program has correctly reserved GPU memory and that it
# actually runs in GPU(s).
#
# Memory usage is in the middle column and GPU usage is in the rightmost co-
# lumn. If GPU usage shows 0% then your code runs only in CPU, not in GPU.
#######################

# To show plots from server, make X11 connection and add this to Run configuration > Environment variables:
# DISPLAY=localhost:10.0

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

    # Opening OVIRS spectra measured from Bennu
    Bennu_path = Path(C.spectral_path, 'Bennu_OVIRS')
    file_list = os.listdir(Bennu_path)

    # Group files by time of day: 20190425 is 3pm data, 20190509 is 12 pm data, 20190516 is 10 am data
    # Empty lists with two elements each for holding the fits data
    Bennu_10 = [None] * 2
    Bennu_12 = [None] * 2
    Bennu_15 = [None] * 2
    for filename in file_list:
        filepath = Path(Bennu_path, filename)
        # Open .fits file with astropy, append to a list

        # A is uncorrected radiance, B is thermal tail removed radiance: make first element the uncorrected
        if 'A' in filename:
            index = 0
        elif 'B' in filename:
            index = 1

        if '20190425' in filename:
            hdulist = fits.open(filepath)
            Bennu_15[index] = hdulist
        elif '20190509' in filename:
            hdulist = fits.open(filepath)
            Bennu_12[index] = hdulist
        elif '20190516' in filename:
            hdulist = fits.open(filepath)
            Bennu_10[index] = hdulist

    def bennu_refine(fitslist):

        uncorrected_fits = fitslist[0]
        corrected_fits = fitslist[1]

        # # Handy info print of what the fits file includes:
        # corrected_fits.info()

        wavelengths = uncorrected_fits[1].data
        # header = corrected_fits[0].header
        uncorrected_rad = uncorrected_fits[0].data[:, 0, :]
        corrected_rad = corrected_fits[0].data[:, 0, :]

        uncor_sum_rad = np.sum(uncorrected_rad, 1)
        cor_sum_rad = np.sum(corrected_rad, 1)

        # Data is from several scans over Bennu's surface, each scan beginning and ending off-asteroid. See plot of
        # radiances summed over wl:s:
        plt.figure()
        plt.plot(range(len(uncor_sum_rad)), uncor_sum_rad)
        plt.xlabel('Measurement number')
        plt.ylabel('Radiance summed over wavelengths')
        plt.show()

        # Go over the summed uncorrected radiances, and save the indices where radiance is over 0.02 (value from plots):
        # gives indices of datapoints where the FOV was on Bennu
        Bennu_indices = []
        index = 0
        for sum_rad in uncor_sum_rad:
            if sum_rad > 0.02:
                Bennu_indices.append(index)
            index = index + 1

        # Pick out the spectra where sum radiance was over threshold value
        uncorrected_Bennu = uncorrected_rad[Bennu_indices, :]
        corrected_Bennu = corrected_rad[Bennu_indices, :]

        plt.figure()
        plt.plot(wavelengths, uncorrected_Bennu[0, :])
        plt.plot(wavelengths, corrected_Bennu[0, :])

        plt.show()

        foo = 0

    bennu_refine(Bennu_15)
    bennu_refine(Bennu_12)
    bennu_refine(Bennu_10)


    # Create wl-vector from 1 to 2.5 µm, with step size in µm
    waves = C.wavelengths

    # #############################
    # # Load meteorite reflectances from files and create more from them through augmentation
    # train_reflectances, test_reflectances = refl.read_meteorites(waves)
    # refl.augmented_reflectances(train_reflectances, waves, test=False)
    # refl.augmented_reflectances(test_reflectances, waves, test=True)
    # #############################

    #############################
    # # Load asteroid reflectances, they are already augmented in a more sophisticated manner
    # train_reflectances, test_reflectances = refl.read_asteroids()

    #############################
    # # Calculate 10 radiances from each reflectance, and save them on disc as toml
    # rad.calculate_radiances(test_reflectances, test=True)
    # rad.calculate_radiances(train_reflectances, test=False)

    # ##############################
    # # Plot uncorrected and (ideally) corrected reflectance from one radiance sample to illustrate why this is relevant
    # rad_dict = tomler.read_radiance('rads_5700.toml')
    # meta = rad_dict['metadata']
    # uncorrected = rad.radiance2reflectance(rad_dict['sum_radiance'], meta['heliocentric_distance'], meta['phase_angle'], meta['emission_angle'])
    # corrected = rad.radiance2reflectance(rad_dict['reflected_radiance'], meta['heliocentric_distance'], meta['phase_angle'], meta['emission_angle'])
    #
    # plt.figure()
    # plt.plot(C.wavelengths, corrected)
    # plt.plot(C.wavelengths, uncorrected)
    # plt.legend(('Corrected', 'Uncorrected'))
    # plt.xlabel('Wavelength [µm]')
    # plt.ylabel('Reflectance')
    # plt.show()
    # ##############################
    # # Create a "bunch" from training and testing radiances and save both in their own files. This is orders of magnitude
    # # faster than reading each radiance from its own toml

    # def bunch_rads(summed, separate, filepath: Path):
    #
    #     rad_bunch = {}
    #     rad_bunch['summed'] = summed
    #     rad_bunch['separate'] = separate
    #
    #     with open(filepath, 'wb') as file_pi:
    #         pickle.dump(rad_bunch, file_pi)
    #
    #
    # summed_test, separate_test = rad.read_radiances(test=True)
    # bunch_rads(summed_test, separate_test, C.rad_bunch_test_path)
    #
    # summed_training, separate_training = rad.read_radiances(test=False)
    # bunch_rads(summed_training, separate_training, C.rad_bunch_training_path)

    # ##############################

    # Load training radiances from one file as dicts
    with open(C.rad_bunch_training_path, 'rb') as file_pi:
        rad_bunch_training = pickle.load(file_pi)

    X_train = rad_bunch_training['summed']
    y_train = rad_bunch_training['separate']

    # model = NN.train_autoencoder(X_train, y_train, early_stop=True, checkpoints=True, save_history=True)
    model = NN.train_autoencoder(X_train, y_train, early_stop=False, checkpoints=True, save_history=True)


    ##############################

    # Load test radiances from one file as dicts
    with open(C.rad_bunch_test_path, 'rb') as file_pi:
        rad_bunch_test = pickle.load(file_pi)

    X_test = rad_bunch_test['summed']
    y_test = rad_bunch_test['separate']



    test_history = model.evaluate(X_test, y_test)

    for i in range(20):
        test_sample = np.expand_dims(X_test[i, :], axis=0)
        prediction = model.predict(test_sample).squeeze() #model.predict(np.array([summed.T])).squeeze()
        pred1 = prediction[0:int(len(prediction) / 2)]
        pred2 = prediction[int(len(prediction) / 2):len(prediction) + 1]

        plt.figure()
        x = waves
        plt.plot(x, y_test[i, :, 0], 'r')
        plt.plot(x, y_test[i, :, 1], 'b')
        plt.plot(x, pred1.squeeze(), '--c')
        plt.plot(x, pred2.squeeze(), '--m')
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Radiance')
        plt.legend(('ground 1', 'ground 2', 'prediction 1', 'prediction 2'))

        fig_filename = C.run_figname + f'_test{i+1}.png'
        fig_path = Path(C.training_path, fig_filename)
        plt.savefig(fig_path, dpi=300)

    # plt.show()
    # print('test')







