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

    # summed, separate = rad.read_radiances()
    #
    # rad_bunch = {}
    # rad_bunch['summed'] = summed
    # rad_bunch['separate'] = separate
    # # tomler.save_rad_bunch(rad_bunch)
    # with open(C.rad_bunch_path, 'wb') as file_pi:
    #     pickle.dump(rad_bunch, file_pi)

    with open(C.rad_bunch_path, 'rb') as file_pi:
        rad_bunch = pickle.load(file_pi)

    X = rad_bunch['summed']
    y = rad_bunch['separate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

    model = NN.train_autoencoder(X_train, y_train, early_stop=False, checkpoints=False, save_history=False)

    for i in range(10):
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

        plt.legend(('ground 1', 'ground 2', 'prediction 1', 'prediction 2'))

    plt.show()
    print('test')

    # Build and train autoencoder

    # Data in the form of:
    # data, ground1, ground2 = create_data(length, samples)
    # ground = np.zeros((samples, length, 2))
    # ground[:, :, 0] = ground1
    # ground[:, :, 1] = ground2












