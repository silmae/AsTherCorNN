import random
from os import path
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
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

# For running with GPU on server:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Check available GPU with command nvidia-smi in terminal
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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

    # Create wl-vector from 1 to 2.5 µm, with step size in µm
    waves = C.wavelengths

    #############################
    # Load meteorite reflectances from files and create more from them through augmentation
    # refl.augmented_reflectances(waves)
    # foo = refl.read_meteorites(waves)
    #############################

    # rad.calculate_radiances()

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

    # summed, separate = rad.read_radiances()

    # rad_bunch = {}
    # rad_bunch['summed'] = summed
    # rad_bunch['separate'] = separate
    # # tomler.save_rad_bunch(rad_bunch)
    # with open(C.rad_bunch_path, 'wb') as file_pi:
    #     pickle.dump(rad_bunch, file_pi)
    #
    # Load radiances from one file as dicts
    with open(C.rad_bunch_path, 'rb') as file_pi:
        rad_bunch = pickle.load(file_pi)

    X = rad_bunch['summed']
    y = rad_bunch['separate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

    model = NN.train_autoencoder(X_train, y_train, early_stop=True, checkpoints=True, save_history=True)

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







