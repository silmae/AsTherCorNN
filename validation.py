"""
Methods for testing neural network performance, using both synthetic data and real data OVIRS data of Bennu.
"""

import time
from pathlib import Path
import os
from contextlib import redirect_stdout  # For saving keras prints into text files

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sklearn.utils

import constants as C
import file_handling
import neural_network as NN
import radiance_data as rad
import file_handling as FH


def test_model(x_test, y_test, model, thermal_radiances, savefolder):
    """
    Testing a trained Keras model with data given as parameters. Saving results of test as a toml and as several plots.

    :param x_test:
        Test data
    :param y_test:
        Ground truth for test data
    :param model:
        A trained Keras model with weights loaded
    :param thermal_radiances:
        Thermal spectral radiances for all test data
    :param savefolder:
        Path to folder where results and plots will be saved

    :return:
        Dictionary containing the calculated errors
    """

    # Run test for the model with Keras, take test result and elapsed time into variables to save them later
    time_start = time.perf_counter_ns()
    test_result = model.evaluate(x_test, y_test, verbose=0)
    time_stop = time.perf_counter_ns()
    elapsed_time_s = (time_stop - time_start) / 1e9
    print(f'Elapsed prediction time for {len(x_test[:, 0])} samples was {elapsed_time_s} seconds')
    print(f'Test with Keras resulted in a loss of {test_result}')

    # Calculate some differences between ground truth and prediction
    # Cosine of angle between two vectors
    def cosine_distance(s1, s2):
        s1_norm = np.sqrt(np.dot(s1, s1))
        s2_norm = np.sqrt(np.dot(s2, s2))
        sum_s1_s2 = np.dot(s1, s2)
        cosangle = (sum_s1_s2 / (s1_norm * s2_norm))
        return cosangle

    # Mean absolute error between two vectors
    def MAE(s1, s2):
        return sum(abs(s1 - s2)) / len(s1)

    # Same as MAE above, but using only the last quarter of the vectors: the errors from thermal tail are more visible
    def tail_MAE(s1, s2):
        s1 = s1[-int(len(s1)/4):]
        s2 = s2[-int(len(s2)/4):]
        error = MAE(s1, s2)
        return error

    # Lists for storing the results of calculations
    reflrad_mae_corrected = []
    reflrad_cos_corrected = []
    reflrad_mae_uncorrected = []
    reflrad_cos_uncorrected = []
    reflectance_mae_corrected = []
    reflectance_cos_corrected = []
    reflectance_mae_uncorrected = []
    reflectance_cos_uncorrected = []
    thermrad_mae = []
    thermrad_cos = []
    temperature_ground = []
    temperature_pred = []
    uncorrected_reflectances = []
    corrected_reflectances = []
    ground_reflectances = []

    indices = range(len(x_test[:, 0]))
    plot_indices = np.random.randint(0, len(x_test[:, 0]), 10)  # Choose 10 random data points for plotting

    for i in indices:
        test_sample = np.expand_dims(x_test[i, :], axis=0)
        prediction = model.predict(test_sample).squeeze()

        pred_temperature = np.asscalar(prediction)
        temperature_pred.append(pred_temperature)
        ground_temperature = y_test[i, 0]
        temperature_ground.append(ground_temperature)

        print(f'Ground temperature: {ground_temperature}')
        print(f'Prediction temperature: {pred_temperature}')

        # Calculate thermal spectral radiance from predicted temperature and constant emissivity using Planck's law
        eps = (C.emissivity_max + C.emissivity_min) / 2  # Mean emissivity in training data
        pred_radiance = rad.thermal_radiance(pred_temperature, eps, C.wavelengths)

        pred_therm = pred_radiance[:, 1]
        ground_therm = thermal_radiances[i, :]

        # Reflected radiance = sum radiance (input) - thermal radiance (from temperature)
        pred_refl = test_sample.squeeze() - pred_therm
        ground_refl = test_sample.squeeze() - ground_therm
        uncorrected_refl = test_sample.squeeze()

        # Plot some results for closer inspection from randomly chosen test spectra
        if i in plot_indices:
            plot_val_test_results(test_sample, ground_refl, ground_therm, pred_refl, pred_therm, savefolder, i+1)

        # Calculate normalized reflectance from uncorrected, NN-corrected, and ground truth reflected radiances
        reflectance_ground = rad.radiance2norm_reflectance(ground_refl)
        ground_reflectances.append(reflectance_ground)
        reflectance_uncorrected = rad.radiance2norm_reflectance(uncorrected_refl)
        uncorrected_reflectances.append(reflectance_uncorrected)
        reflectance_corrected = rad.radiance2norm_reflectance(pred_refl)
        corrected_reflectances.append(reflectance_corrected)

        # Mean absolute errors from radiance
        mae1_corrected = MAE(pred_refl, ground_refl)
        mae1_uncorrected = MAE(uncorrected_refl, ground_refl)
        mae2 = MAE(pred_therm, ground_therm)
        # mae1_corrected = tail_MAE(pred_refl, ground_refl)
        # mae1_uncorrected = tail_MAE(uncorrected_refl, ground_refl)
        # mae2 = tail_MAE(pred_therm, ground_therm)
        reflrad_mae_corrected.append(mae1_corrected)
        reflrad_mae_uncorrected.append(mae1_uncorrected)
        thermrad_mae.append(mae2)

        # Cosine distances from radiances
        cosang1_corrected = cosine_distance(pred_refl, ground_refl)
        cosang1_uncorrected = cosine_distance(uncorrected_refl, ground_refl)
        cosang2 = cosine_distance(pred_therm, ground_therm)
        reflrad_cos_corrected.append(cosang1_corrected)
        reflrad_cos_uncorrected.append(cosang1_uncorrected)
        thermrad_cos.append(cosang2)

        # Mean absolute errors and cosine distance from reflectances
        R_MAE_corrected = MAE(reflectance_corrected, reflectance_ground)
        R_MAE_uncorrected = MAE(reflectance_uncorrected, reflectance_ground)
        # R_MAE_corrected = tail_MAE(reflectance_corrected, reflectance_ground)
        # R_MAE_uncorrected = tail_MAE(reflectance_uncorrected, reflectance_ground)
        reflectance_mae_corrected.append(R_MAE_corrected)
        reflectance_mae_uncorrected.append(R_MAE_uncorrected)

        R_cos_corrected = cosine_distance(reflectance_corrected, reflectance_ground)
        R_cos_uncorrected = cosine_distance(reflectance_uncorrected, reflectance_ground)
        reflectance_cos_corrected.append(R_cos_corrected)
        reflectance_cos_uncorrected.append(R_cos_uncorrected)

        print(f'Calculated MAE and cosine angle for sample {i} out of {len(indices)}')

    # Normalized Root Mean Square Error (NRMSE) from temperature predictions
    temperature_errors = np.asarray(temperature_ground) - np.asarray(temperature_pred)
    temperature_NRMSE = np.sqrt((sum(temperature_errors ** 2)) / len(temperature_ground)) / np.mean(temperature_ground)

    # Calculate mean and std of temperature predictions as function of ground temperature
    temperature_pred_mean_std = _calculate_temperature_pred_mean_and_std(temperature_ground, temperature_pred)

    # Mean reflectance spectra: ground, uncorrected, corrected
    mean_reflectance_ground = np.mean(np.asarray(ground_reflectances), axis=0)
    mean_reflectance_uncorrected = np.mean(np.asarray(uncorrected_reflectances), axis=0)
    mean_reflectance_corrected = np.mean(np.asarray(corrected_reflectances), axis=0)

    # Gather all calculated errors in a single dictionary and save that as toml
    mean_dict = {}
    mean_dict['samples'] = i+1
    mean_dict['NN_test_result'] = test_result
    mean_dict['elapsed_prediction_time_s'] = elapsed_time_s
    mean_dict['temperature_NRMSE'] = temperature_NRMSE
    mean_dict['mean_reflected_MAE'] = np.mean(reflrad_mae_corrected)
    mean_dict['mean_thermal_MAE'] = np.mean(thermrad_mae)
    mean_dict['mean_reflected_SAM'] = np.mean(reflrad_cos_corrected)
    mean_dict['mean_thermal_SAM'] = np.mean(thermrad_cos)
    mean_dict['mean_ground_reflectance'] = mean_reflectance_ground
    mean_dict['mean_uncorrected_reflectance'] = mean_reflectance_uncorrected
    mean_dict['mean_corrected_reflectance'] = mean_reflectance_corrected

    temperature_dict = {}
    temperature_dict['ground_temperature'] = temperature_ground
    temperature_dict['predicted_temperature'] = temperature_pred
    temperature_dict['predicted_temperature_mean_and_std'] = temperature_pred_mean_std

    MAE_dict = {}
    MAE_dict['reflected_MAE'] = reflrad_mae_corrected
    MAE_dict['reflected_MAE_uncorrected'] = reflrad_mae_uncorrected
    MAE_dict['thermal_MAE'] = thermrad_mae
    MAE_dict['reflectance_MAE'] = reflectance_mae_corrected
    MAE_dict['reflectance_MAE_uncorrected'] = reflectance_mae_uncorrected

    SAM_dict = {}
    SAM_dict['reflected_SAM'] = reflrad_cos_corrected
    SAM_dict['reflected_SAM_uncorrected'] = reflrad_cos_uncorrected
    SAM_dict['thermal_SAM'] = thermrad_cos
    SAM_dict['reflectance_SAM'] = reflectance_cos_corrected
    SAM_dict['reflectance_SAM_uncorrected'] = reflectance_cos_uncorrected

    error_dict = {}
    error_dict['mean'] = mean_dict
    error_dict['temperature'] = temperature_dict
    error_dict['MAE'] = MAE_dict
    error_dict['SAM'] = SAM_dict

    FH.save_toml(error_dict, Path(savefolder, 'errors.toml'))
    error_plots(savefolder)

    # Return the dictionary containing calculated errors, in addition to saving it on disc
    return error_dict


def _calculate_temperature_pred_mean_and_std(temperature_ground, temperature_pred):
    # Mean predicted temperature and its std for each unique value of ground temperature
    temperature_ground = np.asarray(temperature_ground)
    temperature_pred = np.asarray(temperature_pred)
    unique_ground_temps = np.unique(temperature_ground)
    temperature_pred_mean_std = np.zeros((3, len(unique_ground_temps)))
    saveindex = 0
    for temp in unique_ground_temps:
        index = np.where(temperature_ground == temp)  # Indices where ground temperature is the one to be calculated
        mean = np.mean(temperature_pred[index])
        std = np.std(temperature_pred[index])
        temperature_pred_mean_std[:, saveindex] = [temp, mean, std]
        saveindex = saveindex + 1

    return temperature_pred_mean_std

def plot_val_test_results(test_sample, ground1, ground2, pred1, pred2, savefolder, index):
    """
    Plotting some results for test with one sample radiance, and saving plots on disc.

    :param test_sample:
        Uncorrected spectral radiance
    :param ground1:
        Ground truth spectrum 1: reflected radiance
    :param ground2:
        Ground truth spectrum 2: thermally emitted radiance
    :param pred1:
        Prediction 1: reflected radiance
    :param pred2:
        Prediction 2: thermally emitted radiance
    :param savefolder:
        Path to folder where plots will be saved
    :param index:
        Index of the plotted sample, used in filename of plots
    """

    fig = plt.figure()
    x = C.wavelengths
    plt.plot(x, ground1)
    plt.plot(x, ground2)
    plt.plot(x, pred1.squeeze(), linestyle='--')
    plt.plot(x, pred2.squeeze(), linestyle='--')
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Radiance [W / m² / sr / µm]')
    plt.legend(('ground 1', 'ground 2', 'prediction 1', 'prediction 2'))

    fig_filename = C.training_run_name + f'_test_{index + 1}_radiance.png'
    fig_path = Path(savefolder, fig_filename)
    plt.savefig(fig_path)
    plt.close(fig)

    # Plot and save reflectances: from uncorrected (test_sample), ground truth (y_test), and NN corrected (pred1)
    uncorrected = rad.radiance2norm_reflectance(test_sample).squeeze()
    ground = rad.radiance2norm_reflectance(ground1)
    NN_corrected = rad.radiance2norm_reflectance(pred1)

    fig = plt.figure()
    x = C.wavelengths
    plt.plot(x, uncorrected, color=C.uncor_plot_color)
    plt.plot(x, ground, color=C.ground_plot_color)
    plt.plot(x, NN_corrected, color=C.NNcor_plot_color)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Normalized reflectance')
    plt.legend(('Uncorrected', 'Ground truth', 'NN-corrected'))

    fig_filename = C.training_run_name + f'_test_{index + 1}_reflectance.png'
    fig_path = Path(savefolder, fig_filename)
    plt.savefig(fig_path)
    plt.close(fig)

    # Plot and save thermal radiances: ground truth and NN-result
    ground = ground2
    NN_corrected = pred2

    fig = plt.figure()
    x = C.wavelengths
    plt.plot(x, ground, color=C.ground_plot_color)
    plt.plot(x, NN_corrected, color=C.NNcor_plot_color)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Thermal radiance [W / m² / sr / µm]')
    plt.legend(('Ground truth', 'NN-corrected'))

    fig_filename = C.training_run_name + f'_test_{index + 1}_thermal.png'
    fig_path = Path(savefolder, fig_filename)
    plt.savefig(fig_path)
    plt.close(fig)


def validate_synthetic(model, validation_run_folder: Path):
    """
    Run tests for trained model with synthetic data similar to training data.

    :param model:
        Trained Keras model with weights loaded
    :param validation_run_folder:
        Path to folder where results will be saved. Method will create a sub-folder inside for results from synthetic.
    """

    # Load test radiances from one file as dicts, separate ground truth and test samples
    rad_bunch_test = FH.load_pickle(C.rad_bunch_test_path)
    x_test = rad_bunch_test['radiances']
    y_test = rad_bunch_test['parameters']

    # Limiting test sample temperatures to stay between the min and max temperatures of Bennu, to get comparable errors
    min_temperature = 303
    max_temperature = 351

    min_indices = np.where(y_test[:, 0] == min_temperature)
    min_index = min_indices[0][0]

    max_indices = np.where(y_test[:, 0] == max_temperature)
    max_index = max_indices[0][-1]

    x_test = x_test[min_index:max_index, :]
    y_test = y_test[min_index:max_index, :]

    # Shuffle the data
    x_test, y_test = sklearn.utils.shuffle(x_test, y_test, random_state=0)

    sample_percentage = 15  # percentage of validation data samples used for error calculation, takes less time
    indices = range(int(len(x_test[:, 0]) * (sample_percentage * 0.01)))
    x_test = x_test[indices]
    y_test = y_test[indices]

    # Calculate thermal spectral radiance for each parameter pair
    thermal_radiances = np.zeros((len(y_test), len(C.wavelengths)))
    for i in range(len(y_test)):
        temperature = y_test[i, 0]
        emissivity = y_test[i, 1]

        thermal_radiance = rad.thermal_radiance(temperature, emissivity, C.wavelengths)
        thermal_radiances[i, :] = thermal_radiance[:, 1]

    validation_plots_synthetic_path = Path(validation_run_folder, 'synthetic_validation')
    if os.path.isdir(validation_plots_synthetic_path) == False:
        os.mkdir(validation_plots_synthetic_path)

    error_dict = test_model(x_test, y_test, model, thermal_radiances, validation_plots_synthetic_path)


def bennu_refine(fitslist: list, time: int, discard_indices, plots=False):
    """
    Refine Bennu data to suit testing. Discard measurements where the instrument did not point to Bennu, but to
    the surrounding void of space. Interpolate spectra to match the wavelengths of the training data. Convert radiance
    values to match the units of training data, from [W/cm²/sr/µm] to [W/m²/sr/µm].

    Discard data-points where the integration of different wavelength range sensors was apparently erroneous: these
    data were picked out by hand based on how the radiance plots looked.

    Return uncorrected spectral radiance, thermal tail subtracted spectral radiance, and the thermal tail spectral
    radiance.

    :param fitslist: list
        List of spectral measurements in FITS format
    :param time: int
        Local time on Bennu where the measurement was taken, can be 1000, 1230, or 1500. Affects save locations
    :param discard_indices
        Indices of datapoints that will be discarded as erroneous
    :param plots: boolean
        Whether plots will be made and saved
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

    # Data is from several scans over Bennu's surface, each scan beginning and ending off-asteroid. See this plot of
    # radiances summed over wl:s to illustrate:
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

    # Some data were marked for discarding due to weird looking spectra with very sharp drops and rises
    # Go through discard indices, remove the listed data from the Bennu_indices
    # Must loop through the list backwards to not modify indices of upcoming elements
    for i in sorted(discard_indices, reverse=True):
        del Bennu_indices[i]

    # Pick out the spectra where sum radiance was over threshold value and data was not marked for discarding
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

    # Fetch NASA's temperature and emissivity predictions from FITS file
    temperature = corrected_fits[3].data[:, 0]
    temperature = temperature[Bennu_indices]
    emissivity = corrected_fits[4].data
    emissivity = emissivity[Bennu_indices]

    if plots==True:
        plotpath = Path(C.bennu_plots_path, str(time))
        for i in range(len(Bennu_indices)):
            fig = plt.figure()
            plt.plot(C.wavelengths, uncorrected_Bennu[i, :])
            plt.plot(C.wavelengths, corrected_Bennu[i, :])
            plt.plot(C.wavelengths, thermal_tail_Bennu[i, :])
            plt.legend(('Uncorrected', 'Corrected', 'Thermal tail'))
            plt.xlabel('Wavelength [µm]')
            plt.ylabel('Radiance [W / m² / sr / µm]')
            plt.xlim((2, 2.45))
            plt.ylim((0, 0.4))
            plt.savefig(Path(plotpath, f'bennurads_{time}_{i}.png'))
            print(f'Saved figure as bennurads_{time}_{i}.png')
            # plt.show()
            plt.close(fig)

    return uncorrected_Bennu, corrected_Bennu, thermal_tail_Bennu, temperature, emissivity


def validate_bennu(model, validation_run_folder):
    """
    Test model using OVIRS data of Bennu from three local times. Load data and refine it to match the training and
    validation data. Saves the results of testing as toml file and some plots drawn from the results.

    :param model:
        Trained Keras model with weights loaded
    :param validation_run_folder:
        Path to the folder where results will be saved
    """

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

    # Load indices that mark data which will be discarded
    discard_1000 = FH.load_csv(Path(Bennu_path, '1000_discard_indices'))
    discard_1230 = FH.load_csv(Path(Bennu_path, '1230_discard_indices'))
    discard_1500 = FH.load_csv(Path(Bennu_path, '1500_discard_indices'))

    # Interpolate to match the wl vector used for training data, convert the readings to another radiance unit
    uncorrected_1500, corrected_1500, thermal_tail_1500, temperatures_1500, emissivities_1500 = bennu_refine(Bennu_1500, 1500, discard_1500, plots=False)
    uncorrected_1230, corrected_1230, thermal_tail_1230, temperatures_1230, emissivities_1230 = bennu_refine(Bennu_1230, 1230, discard_1230, plots=False)
    uncorrected_1000, corrected_1000, thermal_tail_1000, temperatures_1000, emissivities_1000 = bennu_refine(Bennu_1000, 1000, discard_1000, plots=False)

    # # Finding min and max temperatures from the data. 10:00 is the coldest time, 12:30 is the hottest
    # Bennu_min_temperature = min(temperatures_1000)
    # Bennu_max_temperature = max(temperatures_1230)

    # Call test function for a dataset from one local time
    def test_model_Bennu(X_Bennu, temperatures, emissivities, thermal, time: str, validation_plots_Bennu_path):
        # Organize ground truth data to match what the ML model expects
        y_Bennu = np.zeros((len(X_Bennu[:,0]), 2))
        y_Bennu[:, 0] = temperatures
        y_Bennu[:, 1] = emissivities

        savepath = Path(validation_plots_Bennu_path, time)
        if os.path.isdir(savepath) == False:
            os.mkdir(savepath)

        error_dict = test_model(X_Bennu, y_Bennu, model, thermal, savepath)

        return error_dict

    # Save location of plots from validating with Bennu data
    validation_plots_Bennu_path = Path(validation_run_folder, 'bennu_validation')
    if os.path.isdir(validation_plots_Bennu_path) == False:
        os.mkdir(validation_plots_Bennu_path)

    print('Testing with Bennu data, local time 15:00')
    errors_1500 = test_model_Bennu(uncorrected_1500, temperatures_1500, emissivities_1500, thermal_tail_1500, str(1500), validation_plots_Bennu_path)

    print('Testing with Bennu data, local time 12:30')
    errors_1230 = test_model_Bennu(uncorrected_1230, temperatures_1230, emissivities_1230, thermal_tail_1230, str(1230), validation_plots_Bennu_path)

    print('Testing with Bennu data, local time 10:00')
    errors_1000 = test_model_Bennu(uncorrected_1000, temperatures_1000, emissivities_1000, thermal_tail_1000, str(1000), validation_plots_Bennu_path)

    # Collecting calculated results into one dictionary and saving it as toml
    errors_Bennu = {}
    errors_Bennu['errors_1000'] = errors_1000
    errors_Bennu['errors_1230'] = errors_1230
    errors_Bennu['errors_1500'] = errors_1500
    FH.save_toml(errors_Bennu, Path(validation_plots_Bennu_path, 'errors_Bennu.toml'))

    # Plotting errors from all three local times. Plots for individual times are made in the test_model -method
    plot_Bennu_errors(validation_plots_Bennu_path)


def validate_and_test(last_epoch):
    """
    Run validation with synthetic data and testing with real data, for a trained model given as argument.

    :param model: Keras Model -instance
        A trained model that will be tested
    """

    # Generate a unique folder name for results of test based on time the test was run
    timestr = time.strftime("%Y%m%d-%H%M%S")
    timestr = 'test'  # Folder name for test runs, otherwise a new folder is always created

    # Create folder for results
    validation_run_folder = Path(C.val_and_test_path, f'validation-run_epoch-{last_epoch}_time-{timestr}')
    if os.path.isdir(validation_run_folder) == False:
        os.mkdir(validation_run_folder)

    # Build a model and load pre-trained weights
    model = NN.create_model(
        conv_filters=C.conv_filters,
        conv_kernel=C.conv_kernel,
        encoder_start=C.encoder_start,
        encoder_node_relation=C.encoder_node_relation,
        encoder_stop=C.encoder_stop,
        lr=C.learning_rate
    )

    weight_path = Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5')
    model.load_weights(weight_path)

    # Print summary of model architecture into file
    with open(Path(validation_run_folder, 'modelsummary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # Validation with synthetic data similar to training data
    validate_synthetic(model, validation_run_folder)

    # Testing with real asteroid data
    validate_bennu(model, validation_run_folder)

# TODO add to temperature plot mean and std of predtemp as function of groundtemp, using this:

def _plot_with_shadow(ax_obj, x_data, y_data, y_data_std, color, label, ls='-') -> None:
    """ Method by K.A. Riihiaho, copied (with permission) from
    https://github.com/silmae/HyperBlend/blob/master/src/plotter.py

    Plot data with standard deviation as shadow.
    Data must be sorted to show correctly.
    :param ax_obj:
        Pyplot axes object to plot to.
    :param x_data:
        Data x values (wavelengths).
    :param y_data:
        Data y values as numpy.array.
    :param y_data_std:
        Standard deviation as numpy array. The shadow is drawn as +- std/2.
    :param color:
        Color of the plot and shadow.
    :param label:
        Label of the value.
    :param ls:
        Line style. See pyplot linestyle documentation.
    """

    ax_obj.fill_between(x_data, y_data - (y_data_std / 2), y_data + (y_data_std / 2), alpha=0.1,
                        color=color)
    ax_obj.plot(x_data, y_data, color=color, ls=ls, label=label)


def error_plots(folderpath):
    """
    Load a dictionary containing calculated errors from a toml file, make plots of the errors, and save the plots in the
    folder where the toml file was located.

    :param folderpath:
        Path the to folder where the toml file of errors is and where the plots will be saved
    """

    # Load dictionary of errors, saved as a toml
    errordict = FH.load_toml(Path(folderpath, 'errors.toml'))

    # Extract the results of test calculations saved in the dict
    temperature_dict = errordict['temperature']
    temperature_ground = np.asarray(temperature_dict['ground_temperature'])
    temperature_pred = np.asarray(temperature_dict['predicted_temperature'])
    temperature_pred_mean_std = temperature_dict['predicted_temperature_mean_and_std']
    temperature_error = temperature_pred - temperature_ground

    mean_dict = errordict['mean']
    mean_ground_reflectance = mean_dict['mean_ground_reflectance']
    mean_uncorrected_reflectance = mean_dict['mean_uncorrected_reflectance']
    mean_corrected_reflectance = mean_dict['mean_corrected_reflectance']

    MAE_dict = errordict['MAE']
    thermrad_mae = MAE_dict['thermal_MAE']
    reflrad_mae_corrected = MAE_dict['reflected_MAE']
    reflrad_mae_uncorrected = MAE_dict['reflected_MAE_uncorrected']
    reflectance_mae_corrected = MAE_dict['reflectance_MAE']
    reflectance_mae_uncorrected = MAE_dict['reflectance_MAE_uncorrected']

    SAM_dict = errordict['SAM']
    thermrad_cos = SAM_dict['thermal_SAM']
    reflrad_cos_corrected = SAM_dict['reflected_SAM']
    reflrad_cos_uncorrected = SAM_dict['reflected_SAM_uncorrected']
    reflectance_cos_corrected = SAM_dict['reflectance_SAM']
    reflectance_cos_uncorrected = SAM_dict['reflectance_SAM_uncorrected']

    # Mean predicted temperature and its std for each unique value of ground temperature
    unique_ground_temps = np.unique(temperature_ground)
    temperature_pred_mean_std = np.zeros((3, len(unique_ground_temps)))
    saveindex = 0
    for temp in unique_ground_temps:
        index = np.where(temperature_ground == temp)  # Indices where ground temperature is the one to be calculated
        mean = np.mean(temperature_pred[index])
        std = np.std(temperature_pred[index])
        temperature_pred_mean_std[:, saveindex] = [temp, mean, std]
        saveindex = saveindex + 1

    # Min and max temperatures, used to plot a reference line corresponding to ideal result
    min_temperature = int(min(temperature_ground))
    max_temperature = int(max(temperature_ground))

    # Predicted temperature as function of ground truth temperature
    fig = plt.figure()
    plt.scatter(temperature_ground, temperature_pred, alpha=0.1, color=C.NNcor_plot_color)
    plt.xlabel('Ground truth temperature [K]')
    plt.ylabel('Predicted temperature [K]')
    # Plot a reference line with slope 1: ideal result
    plt.plot(range(min_temperature, max_temperature), range(min_temperature, max_temperature), color=C.ideal_result_line_color)
    _plot_with_shadow(plt, temperature_pred_mean_std[0, :], temperature_pred_mean_std[1, :], temperature_pred_mean_std[2, :], color=C.mean_std_temp_color, label='Mean and std')
    plt.ylim(C.temperature_plot_ylim)
    plt.savefig(Path(folderpath, 'predtemp-groundtemp.png'))
    plt.close(fig)

    # Mean reflectances of ground truth, uncorrected, and NN corrected
    fig = plt.figure()
    plt.plot(C.wavelengths, mean_ground_reflectance, color=C.ground_plot_color, label='Ground')
    plt.plot(C.wavelengths, mean_uncorrected_reflectance, color=C.uncor_plot_color, label='Uncorrected')
    plt.plot(C.wavelengths, mean_corrected_reflectance, color=C.NNcor_plot_color, label='NN-corrected')
    plt.legend()
    plt.savefig(Path(folderpath, 'mean_reflectances.png'))
    plt.close(fig)

    def double_plot(uncorrected, corrected, label, filename, limit=(0,0)):
        fig = plt.figure()
        plt.scatter(temperature_ground, uncorrected, alpha=0.1, color=C.uncor_plot_color)
        plt.scatter(temperature_ground, corrected, alpha=0.1, color=C.NNcor_plot_color)
        plt.xlabel('Ground truth temperature [K]')
        plt.ylabel(label)
        if limit != (0, 0):
            plt.ylim(limit)
        leg = plt.legend(('Uncorrected', 'NN-corrected'))
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        plt.savefig(Path(folderpath, f'{filename}.png'))
        plt.close(fig)

    # Cosine distance of reflected radiance from ideally corrected result as function of ground truth temperature,
    # for corrected and uncorrected
    double_plot(reflrad_cos_uncorrected, reflrad_cos_corrected, 'Reflected radiance cosine distance', 'reflrad_SAM_groundtemp')

    # The same as above, but from reflectance instead of reflected radiance
    double_plot(reflectance_cos_uncorrected, reflectance_cos_corrected, 'Reflectance cosine distance', 'reflectance_SAM_groundtemp')

    # Mean absolute error of reflected radiance from both corrected and uncorrected, as function of ground truth temp
    double_plot(reflrad_mae_uncorrected, reflrad_mae_corrected, 'Reflected radiance MAE', 'reflrad_MAE_groundtemp')

    # Same as above, but from reflectance
    double_plot(reflectance_mae_uncorrected, reflectance_mae_corrected, 'Reflectance MAE', 'reflectance_MAE_groundtemp')

    # Cosine distance of estimated thermal radiance from ideal result, as function of ground truth temperature
    fig = plt.figure()
    plt.scatter(temperature_ground, thermrad_cos, alpha=0.1, color=C.NNcor_plot_color)
    plt.xlabel('Ground truth temperature [K]')
    plt.ylabel('Thermal cosine distance')
    plt.savefig(Path(folderpath, 'thermSAM_groundtemp.png'))
    plt.close(fig)

    # Same as above, but mean absolute error instead of cosine
    fig = plt.figure()
    plt.scatter(temperature_ground, thermrad_mae, alpha=0.1, color=C.NNcor_plot_color)
    plt.xlabel('Ground truth temperature [K]')
    plt.ylabel('Thermal MAE')
    plt.savefig(Path(folderpath, 'thermMAE_groundtemp.png'))
    plt.close(fig)

    # def plot_and_save(data, label, filename):
    #     fig = plt.figure()
    #     plt.scatter(temperature_ground, data, alpha=0.1)
    #     plt.xlabel('Ground truth temperature')
    #     # plt.ylim(0,0.02)
    #     plt.ylabel(label)
    #     plt.savefig(Path(folderpath, filename))
    #     plt.show()
    #     plt.close(fig)
    #
    # # plot_and_save(np.asarray(refl_MAE_uncor) - np.asarray(refl_MAE), r'MAE($L_{th}$) improvement', 'refl-MAE-improvement_groundtemp.png')
    # # plot_and_save(np.asarray(R_MAE_uncor) - np.asarray(R_MAE), r'MAE($R$) improvement', 'R-MAE-improvement_groundtemp.png')
    #
    # plot_and_save(np.asarray(refl_SAM_uncor) - np.asarray(refl_SAM), r'SAM($L_{th}$) improvement', 'refl-SAM-improvement_groundtemp.png')
    # plot_and_save(np.asarray(R_SAM_uncor) - np.asarray(R_SAM), r'SAM($R$) improvement', 'R-SAM-improvement_groundtemp.png')


def plot_Bennu_errors(folderpath):
    """
    Plot errors using test results from all three local times on Bennu. Loads a dictionary of errors from a toml file,
    and saves the plot in the folder where the toml is located.

    :param folderpath:
        Path to the folder where the error dictionary is, and where the plots will be saved
    """

    # Load error dictionary and pick out the three dictionaries of errors for different local times
    errordict = file_handling.load_toml(Path(folderpath, 'errors_Bennu.toml'))
    errors_1000 = errordict['errors_1000']
    errors_1230 = errordict['errors_1230']
    errors_1500 = errordict['errors_1500']

    # Plotting and saving results for all three datasets
    def Bennuplot(errors_1000, errors_1230, errors_1500, data_name, label, savefolder):

        def fetch_data(errordict, data_name):
            temperature_dict = errordict['temperature']
            ground_temps = np.asarray(temperature_dict['ground_temperature'])

            if 'MAE' in data_name:
                data_dict = errordict['MAE']
                data = np.asarray(data_dict[data_name])

            elif 'SAM' in data_name:
                data_dict = errordict['SAM']
                data = data_dict[data_name]

            elif 'temperature' in data_name:
                data_dict = errordict['temperature']
                data = data_dict[data_name]

            return ground_temps, data

        ground_temps_1000, data_1000 = fetch_data(errors_1000, data_name)
        ground_temps_1230, data_1230 = fetch_data(errors_1230, data_name)
        ground_temps_1500, data_1500 = fetch_data(errors_1500, data_name)

        fig = plt.figure()
        plt.scatter(ground_temps_1000, data_1000, alpha=0.1)
        plt.scatter(ground_temps_1230, data_1230, alpha=0.1)
        plt.scatter(ground_temps_1500, data_1500, alpha=0.1)
        plt.xlabel('Ground truth temperature [K]')
        plt.ylabel(label)
        leg = plt.legend(('10:00', '12:30', '15:00'), title='Local time on Bennu')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        if data_name == 'predicted_temperature':
            plt.plot(range(300, 350), range(300, 350), color=C.ideal_result_line_color)  # Plot a reference line with slope 1: ideal result

            ground_temps = errors_1000['temperature']['ground_temperature'] + errors_1230['temperature']['ground_temperature'] + errors_1500['temperature']['ground_temperature']
            pred_temps = errors_1000['temperature']['predicted_temperature'] + errors_1230['temperature']['predicted_temperature'] + errors_1500['temperature']['predicted_temperature']

            mean_and_std = _calculate_temperature_pred_mean_and_std(ground_temps, pred_temps)

            _plot_with_shadow(plt, mean_and_std[0, :], mean_and_std[1, :], mean_and_std[2, :], color=C.mean_std_temp_color, label='Prediction mean and std')

            plt.ylim(C.temperature_plot_ylim)
        plt.savefig(Path(savefolder, f'{data_name}.png'))
        plt.close(fig)

    savefolder = folderpath
    Bennuplot(errors_1000, errors_1230, errors_1500, 'predicted_temperature', 'Predicted temperature [K]', savefolder)
    Bennuplot(errors_1000, errors_1230, errors_1500, 'reflected_MAE', 'Reflected MAE', savefolder)
    Bennuplot(errors_1000, errors_1230, errors_1500, 'thermal_MAE', 'Thermal MAE', savefolder)
    Bennuplot(errors_1000, errors_1230, errors_1500, 'reflected_SAM', 'Reflected SAM', savefolder)
    Bennuplot(errors_1000, errors_1230, errors_1500, 'thermal_SAM', 'Thermal SAM', savefolder)

    # Plots where error of the uncorrected results is shown alongside the corrected
    def Bennu_comparison_plots(corrected_name, uncorrected_name, label, lim=(0, 0)):
        temp_dict_1000 = errors_1000['temperature']
        ground_temps_1000 = np.asarray(temp_dict_1000['ground_temperature'])
        temp_dict_1230 = errors_1230['temperature']
        ground_temps_1230 = np.asarray(temp_dict_1230['ground_temperature'])
        temp_dict_1500 = errors_1500['temperature']
        ground_temps_1500 = np.asarray(temp_dict_1500['ground_temperature'])

        if 'MAE' in corrected_name:
            dict_name = 'MAE'
        elif 'SAM' in corrected_name:
            dict_name = 'SAM'
        dict_1000 = errors_1000[dict_name]
        corrected_1000 = dict_1000[corrected_name]
        uncorrected_1000 = dict_1000[uncorrected_name]
        dict_1230 = errors_1230[dict_name]
        corrected_1230 = dict_1230[corrected_name]
        uncorrected_1230 = dict_1230[uncorrected_name]
        dict_1500 = errors_1500[dict_name]
        corrected_1500 = dict_1500[corrected_name]
        uncorrected_1500 = dict_1500[uncorrected_name]

        fig = plt.figure()
        # Use one color for uncorrected and other for corrected, with their hex codes determined in constants.py
        uncor_scatter1 = plt.scatter(ground_temps_1000, uncorrected_1000, alpha=0.1, color=C.uncor_plot_color)
        uncor_scatter2 = plt.scatter(ground_temps_1230, uncorrected_1230, alpha=0.1, color=C.uncor_plot_color)
        uncor_scatter3 = plt.scatter(ground_temps_1500, uncorrected_1500, alpha=0.1, color=C.uncor_plot_color)

        cor_scatter1 = plt.scatter(ground_temps_1000, corrected_1000, alpha=0.1, color=C.NNcor_plot_color)
        cor_scatter2 = plt.scatter(ground_temps_1230, corrected_1230, alpha=0.1, color=C.NNcor_plot_color)
        cor_scatter3 = plt.scatter(ground_temps_1500, corrected_1500, alpha=0.1, color=C.NNcor_plot_color)

        # If a limit other than (0,0) is given in arguments, use it for limiting the shown y-axis values
        if lim != (0, 0):
            plt.ylim(lim)

        leg = plt.legend([cor_scatter1, uncor_scatter1], ['NN-corrected', 'Uncorrected'])
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        plt.xlabel('Ground truth temperature [K]')
        plt.ylabel(label)
        plt.savefig(Path(savefolder, f'{corrected_name}_{uncorrected_name}.png'))
        plt.close(fig)

    Bennu_comparison_plots('reflected_MAE', 'reflected_MAE_uncorrected', 'Reflected radiance MAE')  # , lim=(0, 0.005))
    Bennu_comparison_plots('reflected_SAM', 'reflected_SAM_uncorrected',
                           'Reflected radiance cosine distance')  # , lim=(0.9999, 1.0))
    Bennu_comparison_plots('reflectance_MAE', 'reflectance_MAE_uncorrected', 'Reflectance MAE')  # , lim=(0, 0.01))
    Bennu_comparison_plots('reflectance_SAM', 'reflectance_SAM_uncorrected',
                           'Reflectance cosine distance')  # , lim=(0.999, 1.0))

    # Plot of mean reflectances:
    mean_ground_reflectance_1000 = errors_1000['mean']['mean_ground_reflectance']
    mean_ground_reflectance_1230 = errors_1230['mean']['mean_ground_reflectance']
    mean_ground_reflectance_1500 = errors_1500['mean']['mean_ground_reflectance']
    mean_ground_reflectance = np.mean(np.asarray([mean_ground_reflectance_1000, mean_ground_reflectance_1230, mean_ground_reflectance_1500]), axis=0)

    mean_uncorrected_reflectance_1000 = errors_1000['mean']['mean_uncorrected_reflectance']
    mean_uncorrected_reflectance_1230 = errors_1230['mean']['mean_uncorrected_reflectance']
    mean_uncorrected_reflectance_1500 = errors_1500['mean']['mean_uncorrected_reflectance']
    mean_uncorrected_reflectance = np.mean(np.asarray([mean_uncorrected_reflectance_1000, mean_uncorrected_reflectance_1230, mean_uncorrected_reflectance_1500]), axis=0)

    mean_corrected_reflectance_1000 = errors_1000['mean']['mean_corrected_reflectance']
    mean_corrected_reflectance_1230 = errors_1230['mean']['mean_corrected_reflectance']
    mean_corrected_reflectance_1500 = errors_1500['mean']['mean_corrected_reflectance']
    mean_corrected_reflectance = np.mean(np.asarray([mean_corrected_reflectance_1000, mean_corrected_reflectance_1230, mean_corrected_reflectance_1500]), axis=0)

    fig = plt.figure()
    plt.plot(C.wavelengths, mean_ground_reflectance, color=C.ground_plot_color, label='Ground')
    plt.plot(C.wavelengths, mean_uncorrected_reflectance, color=C.uncor_plot_color, label='Uncorrected')
    plt.plot(C.wavelengths, mean_corrected_reflectance, color=C.NNcor_plot_color, label='NN-corrected')
    plt.legend()
    plt.savefig(Path(savefolder, 'mean_reflectances.png'))
    plt.close(fig)

    print('test')



