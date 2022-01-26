import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import pickle

import constants as C
import neural_network as NN
import radiance_data as rad


def test_model(X_test, y_test, model, savefolder):

    test_result = model.evaluate(X_test, y_test, verbose=0)

    with open(C.training_history_path, 'rb') as file_pi:
        train_history = pickle.load(file_pi)
    val_loss = train_history['val_loss']
    val_loss = val_loss[99]  # TODO Replace hardcoded index somehow

    print(f'Test resulted in a loss of {test_result}')
    print(f'Validation loss for model in the last training epoch was {val_loss}')

    for i in range(20):
        # Plot and save radiances from ground truth and radiances produced by the model prediction
        test_sample = np.expand_dims(X_test[i, :], axis=0)
        prediction = model.predict(test_sample).squeeze()  # model.predict(np.array([summed.T])).squeeze()
        pred1 = prediction[0:int(len(prediction) / 2)]
        pred2 = prediction[int(len(prediction) / 2):len(prediction) + 1]

        fig = plt.figure()
        # plt.style.use('seaborn')
        x = C.wavelengths
        plt.plot(x, y_test[i, :, 0])#, color='r')
        plt.plot(x, y_test[i, :, 1])#, color='b')
        plt.plot(x, pred1.squeeze(), linestyle='--')#, '--c')
        plt.plot(x, pred2.squeeze(), linestyle='--')#, '--m')
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Radiance')
        plt.legend(('ground 1', 'ground 2', 'prediction 1', 'prediction 2'))

        fig_filename = C.training_run_name + f'_test_{i + 1}_radiance.png'
        fig_path = Path(savefolder, fig_filename)
        plt.savefig(fig_path)
        plt.close(fig)

        # Plot and save reflectances: from uncorrected (test_sample), ground truth (y_test), and NN corrected (pred1)
        uncorrected = rad.radiance2norm_reflectance(test_sample).squeeze()
        ground = rad.radiance2norm_reflectance(y_test[i, :, 0])
        NN_corrected = rad.radiance2norm_reflectance(pred1)

        fig = plt.figure()
        x = C.wavelengths
        plt.plot(x, uncorrected)
        plt.plot(x, ground)
        plt.plot(x, NN_corrected)
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Normalized reflectance')
        plt.legend(('Uncorrected', 'Ground truth', 'NN-corrected'))

        fig_filename = C.training_run_name + f'_test_{i + 1}_reflectance.png'
        fig_path = Path(savefolder, fig_filename)
        plt.savefig(fig_path)
        plt.close(fig)

        # Plot and save thermal radiances: ground truth and NN-result
        ground = y_test[i, :, 1].squeeze()
        NN_corrected = pred2

        fig = plt.figure()
        x = C.wavelengths
        plt.plot(x, ground)
        plt.plot(x, NN_corrected)
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Thermal radiance')
        plt.legend(('Ground truth', 'NN-corrected'))

        fig_filename = C.training_run_name + f'_test_{i + 1}_thermal.png'
        fig_path = Path(savefolder, fig_filename)
        plt.savefig(fig_path)
        plt.close(fig)

    # plt.show()
    # print('test')

def validate_synthetic(model):
    # Load test radiances from one file as dicts
    with open(C.rad_bunch_test_path, 'rb') as file_pi:
        rad_bunch_test = pickle.load(file_pi)

    X_test = rad_bunch_test['summed']
    y_test = rad_bunch_test['separate']

    test_model(X_test, y_test, model, C.validation_plots_synthetic_path)


def bennu_refine(fitslist, time, plots=False):

    uncorrected_fits = fitslist[0]
    corrected_fits = fitslist[1]

    # # Handy info print of what the fits file includes:
    # corrected_fits.info()

    wavelengths = uncorrected_fits[1].data
    # header = corrected_fits[0].header
    uncorrected_rad = uncorrected_fits[0].data[:, 0, :]
    corrected_rad = corrected_fits[0].data[:, 0, :]
    thermal_tail_rad = corrected_fits[2].data[:, 0, :]

    uncor_sum_rad = np.sum(uncorrected_rad, 1)
    cor_sum_rad = np.sum(corrected_rad, 1)

    # Data is from several scans over Bennu's surface, each scan beginning and ending off-asteroid. See plot of
    # radiances summed over wl:s:
    # plt.figure()
    # plt.plot(range(len(uncor_sum_rad)), uncor_sum_rad)
    # plt.xlabel('Measurement number')
    # plt.ylabel('Radiance summed over wavelengths')
    # plt.show()

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
    thermal_tail_Bennu = thermal_tail_rad[Bennu_indices, :]

    def bennu_rad_interpolation(data, waves):
        # Interpolate the Bennu data to match wl range used in training
        interp_data = np.zeros((len(data[:, 0]), len(C.wavelengths)))
        for i in range(len(data[:, 0])):
            interp_data[i] = np.interp(C.wavelengths, waves, data[i, :])
        return interp_data

    uncorrected_Bennu = bennu_rad_interpolation(uncorrected_Bennu, wavelengths)
    corrected_Bennu = bennu_rad_interpolation(corrected_Bennu, wavelengths)
    thermal_tail_Bennu = bennu_rad_interpolation(thermal_tail_Bennu, wavelengths)

    def rad_unit_conversion(data):
        # Convert from NASAs radiance unit [W/cm²/sr/µm] to my [W/m²/sr/µm]
        converted = data * 10000
        return converted

    uncorrected_Bennu = rad_unit_conversion(uncorrected_Bennu)
    corrected_Bennu = rad_unit_conversion(corrected_Bennu)
    thermal_tail_Bennu = rad_unit_conversion(thermal_tail_Bennu)

    if plots==True:
        plotpath = Path(C.bennu_plots_path, str(time))
        for i in range(len(Bennu_indices)):
            fig = plt.figure()
            plt.plot(C.wavelengths, uncorrected_Bennu[i, :])
            plt.plot(C.wavelengths, corrected_Bennu[i, :])
            plt.plot(C.wavelengths, thermal_tail_Bennu[i, :])
            plt.legend(('Uncorrected', 'Corrected', 'Thermal tail'))
            plt.xlabel('Wavelength [µm]')
            plt.ylabel('Radiance [W/m²/sr/µm]')
            plt.xlim((2, 2.45))
            plt.ylim((0, 0.4))
            # plt.savefig(Path(plotpath, f'bennurads_{time}_{i}.png'))
            print(f'Saved figure as bennurads_{time}_{i}.png')
            plt.show()
            plt.close(fig)
    return uncorrected_Bennu, corrected_Bennu, thermal_tail_Bennu


def validate_bennu(model):
    # Opening OVIRS spectra measured from Bennu
    Bennu_path = Path(C.spectral_path, 'Bennu_OVIRS')
    file_list = os.listdir(Bennu_path)

    # Group files by time of day: 20190425 is 3pm data, 20190509 is 12:30 pm data, 20190516 is 10 am data
    # Empty lists with two elements each for holding the fits data
    Bennu_1000 = [None] * 2
    Bennu_1230 = [None] * 2
    Bennu_1500 = [None] * 2
    for filename in file_list:
        filepath = Path(Bennu_path, filename)
        # Open .fits file with astropy, append to a list according to local time and correction
        # A is uncorrected radiance, B is thermal tail removed radiance: make first element the uncorrected
        if 'A' in filename:
            index = 0
        elif 'B' in filename:
            index = 1

        if '20190425' in filename:
            hdulist = fits.open(filepath)
            Bennu_1500[index] = hdulist
        elif '20190509' in filename:
            hdulist = fits.open(filepath)
            Bennu_1230[index] = hdulist
        elif '20190516' in filename:
            hdulist = fits.open(filepath)
            Bennu_1000[index] = hdulist

    # Interpolate to match the wl vector used for training data, convert the readings to another radiance unit
    uncorrected_1500, corrected_1500, thermal_tail_1500 = bennu_refine(Bennu_1500, 1500, plots=False)
    uncorrected_1230, corrected_1230, thermal_tail_1230 = bennu_refine(Bennu_1230, 1230, plots=False)
    uncorrected_1000, corrected_1000, thermal_tail_1000 = bennu_refine(Bennu_1000, 1000, plots=False)

    def test_model_Bennu(X_Bennu, reflected, thermal, time: str):
        # Organize data to match what the ML model expects
        y_Bennu = np.zeros((len(X_Bennu[:,0]), len(C.wavelengths), 2))
        y_Bennu[:, :, 0] = reflected
        y_Bennu[:, :, 1] = thermal

        test_model(X_Bennu, y_Bennu, model, Path(C.validation_plots_Bennu_path, time))
        # testhist = model.evaluate(X_Bennu, y_Bennu)

    print('Testing with Bennu data, local time 15:00')
    test_model_Bennu(uncorrected_1500, corrected_1500, thermal_tail_1500, str(1500))
    print('Testing with Bennu data, local time 12:30')
    test_model_Bennu(uncorrected_1230, corrected_1230, thermal_tail_1230, str(1230))
    print('Testing with Bennu data, local time 10:00')
    test_model_Bennu(uncorrected_1000, corrected_1000, thermal_tail_1000, str(1000))
