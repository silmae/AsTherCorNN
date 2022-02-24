import time

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import pickle
import symfit
import sympy

import constants as C
import neural_network as NN
import radiance_data as rad
import file_handling as FH


def fit_Planck(radiance: np.ndarray):
    """
    Fit Planck's law to thermal radiance vector, to evaluate the temperature of the radiation source. Approximates
    emittance as constant 0.9.

    :param radiance: np.ndarray
        Thermally emitted spectral radiance
    :return: float
        Temperature of the radiance source, in Kelvin
    """

    # Parametrize Planck's law in simpler terms to make fitting easier
    # Define constants
    # c = speed of light in vacuum, m / s
    # kB = Boltzmann constant, m² kg / s² / K (= J / K)
    # h = Planck constant, m² kg / s (= J s)
    # Planck originally: L_th = eps * (2 * h * c ** 2) / ((wl ** 5) * (exp((h * c) / (wl * k_B * T)) - 1))
    # Lump constants together: a = 2hc² = 1.191e-16 kg m⁴/s³, and b = hc / k_B = 0.01439 m K
    # Re-parametrized version: L_th = eps * a / ((wl ** 5) * (exp(b / (wl * T)) - 1))

    # Values for a and b calculated with Wolfram Alpha
    # Move from m to µm to make computing easier: values are not so extremely small
    a = 1.191e-16  # kg m⁴/s³
    a = a * 1e24  # Convert to kg µm⁴/s³
    b = 0.01439  # m K
    b = b * 1e6  # Convert to µm K
    eps = C.emittance  # Emittance

    # Temperature as fitting parameter, wavelength as variable
    init_guess = (C.T_max + C.T_min) / 2
    T = symfit.Parameter('T', value=init_guess, min=C.T_min, max=C.T_max)
    wl = symfit.Variable('wl')

    model = eps * a / ((wl ** 5) * (sympy.exp(b / (wl * T)) - 1))  # Apply re-parametrized Planck's law

    fit = symfit.Fit(model, C.wavelengths, radiance, minimizer=[symfit.core.minimizers.NelderMead])
    fit_result = fit.execute()

    # Plot of fit result together with original radiance data
    fig = plt.figure()
    y = model(wl=C.wavelengths, T=fit_result.value(T))
    plt.plot(C.wavelengths, y)
    plt.plot(C.wavelengths, radiance)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Radiance [W / m² / sr / µm]')
    # plt.show()
    plt.close(fig)

    temperature = fit_result.value(T)

    return temperature

def test_model(X_test, y_test, model, test_epoch, savefolder):

    time_start = time.perf_counter_ns()
    test_result = model.evaluate(X_test, y_test, verbose=0)
    time_stop = time.perf_counter_ns()
    print(f'Elapsed prediction time in seconds for {len(X_test[:, 0])} samples was {(time_stop - time_start) / 1e9}')
    print(f'Test with Keras resulted in a loss of {test_result}')

    # Calculate some differences between ground truth and prediction vectors
    # Cosine of angle between two vectors
    def cosine_distance(s1, s2):
        s1_norm = np.sqrt(np.dot(s1, s1))
        s2_norm = np.sqrt(np.dot(s2, s2))
        sum_s1_s2 = np.dot(s1, s2)
        cosangle = (sum_s1_s2 / (s1_norm * s2_norm))
        return cosangle
    # Lists for storing the errors
    refl_mae = []
    refl_cos = []
    therm_mae = []
    therm_cos = []
    temperature_error = []
    temperature_ground = []
    temperature_pred = []
    # indices = range(len(X_test[:, 0]))  # Full error calculation, takes some time
    indices = range(int(len(X_test[:, 0]) * 0.1))  # 10 percent of samples used for error calculation, takes less time

    for i in indices:
        test_sample = np.expand_dims(X_test[i, :], axis=0)
        prediction = model.predict(test_sample).squeeze()  # model.predict(np.array([summed.T])).squeeze()
        pred1 = prediction[0:int(len(prediction) / 2)]
        pred2 = prediction[int(len(prediction) / 2):len(prediction) + 1]
        ground1 = y_test[i, :, 0]
        ground2 = y_test[i, :, 1]

        # Calculate temperatures of ground and prediction by fitting to Planck function
        # TODO Create a proper version for Bennu data, NASA provided temperatures in the FITS extension
        ground_temp = fit_Planck(ground2)
        print(f'Ground temperature: {ground_temp}')
        temperature_ground.append(ground_temp)
        pred_temp = fit_Planck(pred2)
        print(f'Prediction temperature: {pred_temp}')
        temperature_pred.append(pred_temp)

        temperature_error.append(ground_temp - pred_temp)

        # Mean absolute error
        mae1 = sum(abs(pred1 - ground1)) / len(pred1)
        mae2 = sum(abs(pred2 - ground2)) / len(pred2)
        refl_mae.append(mae1)
        therm_mae.append(mae2)

        cosang1 = cosine_distance(pred1, ground1)
        cosang2 = cosine_distance(pred2, ground2)
        refl_cos.append(cosang1)
        therm_cos.append(cosang2)

        print(f'Calculated MAE and cosine angle for sample {i} out of {len(indices)}')

    mean_dict = {}
    mean_dict['mean_reflected_MAE'] = np.mean(refl_mae)
    mean_dict['mean_thermal_MAE'] = np.mean(therm_mae)
    mean_dict['mean_reflected_SAM'] = np.mean(refl_cos)
    mean_dict['mean_thermal_SAM'] = np.mean(therm_cos)
    mean_dict['mean_temp_error'] = np.mean(temperature_error)
    mean_dict['samples'] = i+1
    temperature_dict = {}
    temperature_dict['ground_temperature'] = temperature_ground
    temperature_dict['predicted_temperature'] = temperature_pred
    MAE_dict = {}
    MAE_dict['reflected_MAE'] = refl_mae
    MAE_dict['thermal_MAE'] = therm_mae
    SAM_dict = {}
    SAM_dict['reflected_SAM'] = refl_cos
    SAM_dict['thermal_SAM'] = therm_cos
    error_dict = {}
    error_dict['mean'] = mean_dict
    error_dict['temperature'] = temperature_dict
    error_dict['MAE'] = MAE_dict
    error_dict['SAM'] = SAM_dict

    FH.save_toml(error_dict, Path(savefolder, 'errors.toml'))

    # Plot MAE and SAM of all test samples, for both thermal and reflected
    fig = plt.figure()
    plt.plot(range(len(refl_mae)), refl_mae)
    plt.plot(range(len(refl_mae)), therm_mae)
    plt.ylabel('mean absolute error')
    plt.legend(('reflected', 'thermal'))
    plt.savefig(Path(savefolder, 'MAE.png'))
    plt.close(fig)

    fig = plt.figure()
    plt.plot(range(len(refl_cos)), therm_cos)
    plt.plot(range(len(refl_cos)), refl_cos)
    plt.ylabel('cosine distance')
    plt.legend(('thermal', 'reflected'))
    plt.savefig(Path(savefolder, 'SAM.png'))
    plt.close(fig)

    # Plot scatters of ground temperature vs temperature error, thermal MAE and SAM vs height of thermal tail
    fig = plt.figure()
    plt.scatter(temperature_ground, temperature_error, alpha=0.1)
    plt.scatter(temperature_ground, temperature_pred, alpha=0.1)
    plt.legend(('Error', 'Predicted'))
    plt.xlabel('Ground truth temperature')
    plt.savefig(Path(C.figfolder, 'tempscatter.png'))
    plt.close(fig)



    # Plot some results for closer inspection from 25 random test spectra
    index = np.random.randint(0, len(X_test[:, 0]), size=25)
    for i in index:
        # Plot and save some radiances from ground truth and radiances produced by the model prediction
        test_sample = np.expand_dims(X_test[i, :], axis=0)
        prediction = model.predict(test_sample).squeeze()
        pred1 = prediction[0:int(len(prediction) / 2)]
        pred2 = prediction[int(len(prediction) / 2):len(prediction) + 1]

        fig = plt.figure()
        x = C.wavelengths
        plt.plot(x, y_test[i, :, 0])
        plt.plot(x, y_test[i, :, 1])
        plt.plot(x, pred1.squeeze(), linestyle='--')
        plt.plot(x, pred2.squeeze(), linestyle='--')
        plt.xlabel('Wavelength [µm]')
        plt.ylabel('Radiance [W / m² / sr / µm]')
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


def validate_synthetic(model, last_epoch, validation_run_folder):
    # Load test radiances from one file as dicts, separate ground truth and test samples
    rad_bunch_test = FH.load_pickle(C.rad_bunch_test_path)
    X_test = rad_bunch_test['summed']
    y_test = rad_bunch_test['separate']

    validation_plots_synthetic_path = Path(validation_run_folder, 'synthetic_validation')
    os.mkdir(validation_plots_synthetic_path)

    test_model(X_test, y_test, model, last_epoch, validation_plots_synthetic_path)


def bennu_refine(fitslist: list, time: int, plots=False):
    """
    Refine Bennu data to suit testing. Discard measurements where the instrument did not point to Bennu, but to
    the surrounding void of space. Interpolate spectra to match the wavelengths of the training data. Convert radiance
    values to match the units of training data, from [W/cm²/sr/µm] to [W/m²/sr/µm].

    Return uncorrected spectral radiance, thermal tail subtracted spectral radiance, and the thermal tail spectral
    radiance.

    :param fitslist: list
        List of spectral measurements in FITS format
    :param time: int
        Local time on Bennu where the measurement was taken, can be 1000, 1230, or 1500. Affects save locations
    :param plots: boolean
        Whether or not plots will be made and saved
    :return: uncorrected_Bennu, corrected_Bennu, thermal_tail_Bennu:
    """

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
        # Convert from NASAs radiance unit [W/cm²/sr/µm] to [W/m²/sr/µm]
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


def validate_bennu(model, last_epoch, validation_run_folder):
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

    def test_model_Bennu(X_Bennu, reflected, thermal, time: str, validation_plots_Bennu_path):
        # Organize data to match what the ML model expects
        y_Bennu = np.zeros((len(X_Bennu[:,0]), len(C.wavelengths), 2))
        y_Bennu[:, :, 0] = reflected
        y_Bennu[:, :, 1] = thermal

        savepath = Path(validation_plots_Bennu_path, time)
        os.mkdir(savepath)

        test_model(X_Bennu, y_Bennu, model, last_epoch, savepath)
        # testhist = model.evaluate(X_Bennu, y_Bennu)

    validation_plots_Bennu_path = Path(validation_run_folder,
                                       'bennu_validation')  # Save location of plots from validating with Bennu data
    os.mkdir(validation_plots_Bennu_path)

    print('Testing with Bennu data, local time 15:00')
    test_model_Bennu(uncorrected_1500, corrected_1500, thermal_tail_1500, str(1500), validation_plots_Bennu_path)
    print('Testing with Bennu data, local time 12:30')
    test_model_Bennu(uncorrected_1230, corrected_1230, thermal_tail_1230, str(1230), validation_plots_Bennu_path)
    print('Testing with Bennu data, local time 10:00')
    test_model_Bennu(uncorrected_1000, corrected_1000, thermal_tail_1000, str(1000), validation_plots_Bennu_path)


def error_plots(folderpath):
    errordict = FH.load_toml(Path(folderpath, 'errors.toml'))
    # Plot scatters of ground temperature vs temperature error, thermal MAE and SAM vs height of thermal tail
    temperature_dict = errordict['temperature']
    temperature_ground = np.asarray(temperature_dict['ground_temperature'])
    temperature_pred = np.asarray(temperature_dict['predicted_temperature'])
    temperature_error = temperature_pred - temperature_ground

    MAE_dict = errordict['MAE']
    therm_MAE = MAE_dict['thermal_MAE']
    refl_MAE = MAE_dict['reflected_MAE']

    SAM_dict = errordict['SAM']
    therm_SAM = SAM_dict['thermal_SAM']
    refl_SAM = SAM_dict['reflected_SAM']

    fig = plt.figure()
    plt.figure()
    plt.scatter(temperature_ground, temperature_error, alpha=0.1)
    plt.xlabel('Ground truth temperature')
    plt.ylabel('Temperature difference')
    plt.savefig(Path(folderpath, 'tempdif_groundtemp.png'))
    plt.close(fig)

    fig = plt.figure()
    plt.figure()
    plt.scatter(temperature_ground, refl_SAM, alpha=0.1)
    plt.xlabel('Ground truth temperature')
    plt.ylabel('Reflected SAM')
    plt.savefig(Path(folderpath, 'reflSAM_groundtemp.png'))
    plt.close(fig)

    fig = plt.figure()
    plt.figure()
    plt.scatter(temperature_ground, therm_SAM, alpha=0.1)
    plt.xlabel('Ground truth temperature')
    plt.ylabel('Thermal SAM')
    plt.savefig(Path(folderpath, 'thermSAM_groundtemp.png'))
    plt.close(fig)

    fig = plt.figure()
    plt.figure()
    plt.scatter(temperature_ground, therm_MAE, alpha=0.1)
    plt.xlabel('Ground truth temperature')
    plt.ylabel('Thermal MAE')
    plt.savefig(Path(folderpath, 'thermMAE_groundtemp.png'))
    plt.close(fig)

    fig = plt.figure()
    plt.figure()
    plt.scatter(temperature_ground, refl_MAE, alpha=0.1)
    plt.xlabel('Ground truth temperature')
    plt.ylabel('Reflected MAE')
    plt.savefig(Path(folderpath, 'reflMAE_groundtemp.png'))
    plt.close(fig)
