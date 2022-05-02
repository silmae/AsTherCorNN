import time
from pathlib import Path
import os
from contextlib import redirect_stdout  # For saving keras prints into text files

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sklearn.utils
import pickle
import symfit
import sympy

import constants as C
import file_handling
import neural_network as NN
import radiance_data as rad
import file_handling as FH


def fit_Planck(radiance: np.ndarray):
    """
    Fit Planck's law to thermal radiance vector, to evaluate the temperature and emissivity of the radiation source.

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
    # eps = C.emissivity  # Emittance

    # Temperature and emissivity as fitting parameters, wavelength as variable
    init_guess_T = (C.T_max + C.T_min) / 2
    T = symfit.Parameter('T', value=init_guess_T, min=C.T_min, max=C.T_max)
    init_guess_eps = (C.emissivity_max + C.emissivity_min) / 2
    eps = symfit.Parameter('eps', value=init_guess_eps, min=C.emissivity_min, max=C.emissivity_max)
    wl = symfit.Variable('wl')
    C.emissivity_min

    model = eps * a / ((wl ** 5) * (sympy.exp(b / (wl * T)) - 1))  # Apply re-parametrized Planck's law

    fit = symfit.Fit(model, C.wavelengths, radiance, minimizer=[symfit.core.minimizers.NelderMead])
    fit_result = fit.execute()
    # print(f'Emissivity: {fit_result.value(eps)}')

    # # Plot of fit result together with original radiance data
    # fig = plt.figure()
    # y = model(wl=C.wavelengths, T=fit_result.value(T), eps=fit_result.value(eps))
    # plt.plot(C.wavelengths, y)
    # plt.plot(C.wavelengths, radiance)
    # plt.xlabel('Wavelength [µm]')
    # plt.ylabel('Radiance [W / m² / sr / µm]')
    # plt.show()
    # plt.close(fig)

    temperature = fit_result.value(T)

    return temperature

def test_model(X_test, y_test, model, temperatures, savefolder):

    time_start = time.perf_counter_ns()
    test_result = model.evaluate(X_test, y_test, verbose=0)  # (X_test, y_test[:, :, 1], verbose=0)
    time_stop = time.perf_counter_ns()
    elapsed_time_s = (time_stop - time_start) / 1e9
    print(f'Elapsed prediction time for {len(X_test[:, 0])} samples was {elapsed_time_s} seconds')
    print(f'Test with Keras resulted in a loss of {test_result}')

    # Calculate some differences between ground truth and prediction vectors
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

    def tail_MAE(s1, s2):
        s1 = s1[-int(len(s1)/4):]
        s2 = s2[-int(len(s2)/4):]
        error = MAE(s1, s2)
        return error

    # Lists for storing the errors
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
    indices = range(len(X_test[:, 0]))  # Full error calculation, takes some time
    plot_indices = np.random.randint(0, len(X_test[:, 0]), 20)

    for i in indices:
        test_sample = np.expand_dims(X_test[i, :], axis=0)
        prediction = model.predict(test_sample).squeeze()  # model.predict(np.array([summed.T])).squeeze()
        pred_refl = test_sample.squeeze() - prediction
        uncorrected_refl = test_sample.squeeze()
        pred_therm = prediction

        # ground_refl = y_test[i, :, 0]
        ground_therm = y_test[i, :, 1]
        ground_refl = test_sample.squeeze() - ground_therm  # Alternative ground truth to which the alternative reflected is compared

        # Plot some results for closer inspection from 25 random test spectra
        if i in plot_indices:
            plot_val_test_results(test_sample, ground_refl, ground_therm, pred_refl, pred_therm, savefolder, i+1)

        # Calculate normalized reflectance from uncorrected, NN-corrected, and ground truth reflected radiances
        reflectance_ground = rad.radiance2norm_reflectance(ground_refl)
        reflectance_uncorrected = rad.radiance2norm_reflectance(uncorrected_refl)
        reflectance_corrected = rad.radiance2norm_reflectance(pred_refl)

        # Calculate temperature of prediction by fitting to Planck function, compare to ground truth gotten as argument
        ground_temp = temperatures[i]
        print(f'Ground temperature: {ground_temp}')
        temperature_ground.append(ground_temp)
        pred_temp = fit_Planck(pred_therm)
        print(f'Prediction temperature: {pred_temp}')
        temperature_pred.append(pred_temp)

        # Mean absolute errors from radiance
        # mae1_corrected = MAE(pred_refl, ground_refl)
        # mae1_uncorrected = MAE(uncorrected_refl, ground_refl)
        # mae2 = MAE(pred_therm, ground_therm)
        mae1_corrected = tail_MAE(pred_refl, ground_refl)
        mae1_uncorrected = tail_MAE(uncorrected_refl, ground_refl)
        mae2 = tail_MAE(pred_therm, ground_therm)
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
        # R_MAE_corrected = MAE(reflectance_corrected, reflectance_ground)
        # R_MAE_uncorrected = MAE(reflectance_uncorrected, reflectance_ground)
        R_MAE_corrected = tail_MAE(reflectance_corrected, reflectance_ground)
        R_MAE_uncorrected = tail_MAE(reflectance_uncorrected, reflectance_ground)
        reflectance_mae_corrected.append(R_MAE_corrected)
        reflectance_mae_uncorrected.append(R_MAE_uncorrected)

        R_cos_corrected = cosine_distance(reflectance_corrected, reflectance_ground)
        R_cos_uncorrected = cosine_distance(reflectance_uncorrected, reflectance_ground)
        reflectance_cos_corrected.append(R_cos_corrected)
        reflectance_cos_uncorrected.append(R_cos_uncorrected)

        print(f'Calculated MAE and cosine angle for sample {i} out of {len(indices)}')

    # Gather all calculated errors in a single dictionary and save that as toml
    mean_dict = {}
    mean_dict['mean_reflected_MAE'] = np.mean(reflrad_mae_corrected)
    mean_dict['mean_thermal_MAE'] = np.mean(thermrad_mae)
    mean_dict['mean_reflected_SAM'] = np.mean(reflrad_cos_corrected)
    mean_dict['mean_thermal_SAM'] = np.mean(thermrad_cos)
    mean_dict['samples'] = i+1
    mean_dict['elapsed_prediction_time_s'] = elapsed_time_s

    temperature_dict = {}
    temperature_dict['ground_temperature'] = temperature_ground
    temperature_dict['predicted_temperature'] = temperature_pred

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

    # Plot scatters of several errors vs ground truth temperature
    fig = plt.figure()
    plt.scatter(temperature_ground, temperature_pred, alpha=0.1)
    plt.xlabel('Ground truth temperature [K]')
    plt.ylabel('Predicted temperature [K]')
    plt.plot(range(C.T_min, C.T_max), range(C.T_min, C.T_max), 'r')  # Plot a reference line with slope 1: ideal result
    plt.savefig(Path(savefolder, 'predtemp-groundtemp.png'))
    plt.close(fig)

    fig = plt.figure()
    plt.figure()
    plt.scatter(temperature_ground, reflrad_cos_uncorrected, alpha=0.1)
    plt.scatter(temperature_ground, reflrad_cos_corrected, alpha=0.1)
    plt.xlabel('Ground truth temperature [K]')
    plt.ylabel('Reflected cosine distance')
    leg = plt.legend(('Uncorrected', 'NN-corrected'))
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.savefig(Path(savefolder, 'reflrad_SAM_groundtemp.png'))
    plt.close(fig)

    fig = plt.figure()
    plt.figure()
    plt.scatter(temperature_ground, reflectance_cos_uncorrected, alpha=0.1)
    plt.scatter(temperature_ground, reflectance_cos_corrected, alpha=0.1)
    plt.xlabel('Ground truth temperature [K]')
    plt.ylabel('Reflectance cosine distance')
    leg = plt.legend(('Uncorrected', 'NN-corrected'))
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.savefig(Path(savefolder, 'reflectance_SAM_groundtemp.png'))
    plt.close(fig)

    fig = plt.figure()
    plt.figure()
    plt.scatter(temperature_ground, reflectance_mae_uncorrected, alpha=0.1)
    plt.scatter(temperature_ground, reflectance_mae_corrected, alpha=0.1)
    plt.xlabel('Ground truth temperature [K]')
    plt.ylabel('Reflectance MAE')
    leg = plt.legend(('Uncorrected', 'NN-corrected'))
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.savefig(Path(savefolder, 'reflectance_MAE_groundtemp.png'))
    plt.close(fig)

    fig = plt.figure()
    plt.figure()
    plt.scatter(temperature_ground, thermrad_cos, alpha=0.1)
    plt.xlabel('Ground truth temperature [K]')
    plt.ylabel('Thermal cosine distance')
    plt.savefig(Path(savefolder, 'thermSAM_groundtemp.png'))
    plt.close(fig)

    fig = plt.figure()
    plt.figure()
    plt.scatter(temperature_ground, thermrad_mae, alpha=0.1)
    plt.xlabel('Ground truth temperature [K]')
    plt.ylabel('Thermal MAE')
    plt.savefig(Path(savefolder, 'thermMAE_groundtemp.png'))
    plt.close(fig)

    fig = plt.figure()
    plt.figure()
    plt.scatter(temperature_ground, reflrad_mae_uncorrected, alpha=0.1)
    plt.scatter(temperature_ground, reflrad_mae_corrected, alpha=0.1)
    plt.xlabel('Ground truth temperature [K]')
    plt.ylabel('Reflected MAE')
    leg = plt.legend(('Uncorrected', 'NN-corrected'))
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.savefig(Path(savefolder, 'reflrad_MAE_groundtemp.png'))
    plt.close(fig)

    # Return the dictionary containing calculated errors, in addition to saving it on disc
    return error_dict


def plot_val_test_results(test_sample, ground1, ground2, pred1, pred2, savefolder, index):

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
    plt.plot(x, uncorrected)
    plt.plot(x, ground)
    plt.plot(x, NN_corrected)
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
    plt.plot(x, ground)
    plt.plot(x, NN_corrected)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Thermal radiance [W / m² / sr / µm]')
    plt.legend(('Ground truth', 'NN-corrected'))

    fig_filename = C.training_run_name + f'_test_{index + 1}_thermal.png'
    fig_path = Path(savefolder, fig_filename)
    plt.savefig(fig_path)
    plt.close(fig)


def validate_synthetic(model, validation_run_folder: Path):
    """
    Run tests for trained model with synthetic data similar to training data. Calculates temperatures for all data
    points by fitting their ground truth thermal tails to the Planck function.

    :param model:
        Trained model
    :param validation_run_folder:
        Path to folder where results will be saved. Method will create a sub-folder inside for results from synthetic.
    """

    # Load test radiances from one file as dicts, separate ground truth and test samples
    rad_bunch_test = FH.load_pickle(C.rad_bunch_test_path)
    X_test = rad_bunch_test['summed']
    y_test = rad_bunch_test['separate']

    # Shuffle to get samples from all temperatures when using part of the data
    X_test, y_test = sklearn.utils.shuffle(X_test, y_test, random_state=0)

    indices = range(int(len(X_test[:, 0]) * 0.01))  # 1 percent of samples used for error calculation, takes less time
    X_test = X_test[indices]
    y_test = y_test[indices]

    validation_plots_synthetic_path = Path(validation_run_folder, 'synthetic_validation')
    if os.path.isdir(validation_plots_synthetic_path) == False:
        os.mkdir(validation_plots_synthetic_path)

    # Calculate ground truth temperatures by fitting all thermal tails to the Planck function
    temperatures = []
    for i in range(len(y_test[:, 0, 1])):
        ground2 = y_test[i, :, 1]

        # Calculate temperatures of ground and prediction by fitting to Planck function
        temp = fit_Planck(ground2)
        temperatures.append(temp)
        print(f'Calculated temperature {i+1} out of {len(y_test[:, 0, 1])}')

    temperatures = np.asarray(temperatures)

    error_dict = test_model(X_test, y_test, model, temperatures, validation_plots_synthetic_path)
    FH.save_toml(error_dict, Path(validation_plots_synthetic_path, 'error_dict.toml'))


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

    # Data is from several scans over Bennu's surface, each scan beginning and ending off-asteroid. See this plot of
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

    # Fetch NASA's temperature prediction from FITS file
    temperature = corrected_fits[3].data[:, 0]
    temperature = temperature[Bennu_indices]
    # emissivity = corrected_fits[4].data
    # emissivity = emissivity[Bennu_indices]

    # # Alternative temperature prediction by fitting thermal tail to Planck function with constant 0.9 emissivity:
    # # results will not be as truthful, but more in line with how the network thinks
    # temperature = []
    # for thermal_radiance in thermal_tail_Bennu:
    #     temp = fit_Planck(thermal_radiance)
    #     temperature.append(temp)
    # temperature = np.asarray(temperature)

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
            # plt.savefig(Path(plotpath, f'bennurads_{time}_{i}.png'))
            print(f'Saved figure as bennurads_{time}_{i}.png')
            plt.show()
            plt.close(fig)

    return uncorrected_Bennu, corrected_Bennu, thermal_tail_Bennu, temperature


def validate_bennu(model, validation_run_folder):
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
    uncorrected_1500, corrected_1500, thermal_tail_1500, temperatures_1500 = bennu_refine(Bennu_1500, 1500, discard_1500, plots=False)
    uncorrected_1230, corrected_1230, thermal_tail_1230, temperatures_1230 = bennu_refine(Bennu_1230, 1230, discard_1230, plots=False)
    uncorrected_1000, corrected_1000, thermal_tail_1000, temperatures_1000 = bennu_refine(Bennu_1000, 1000, discard_1000, plots=False)

    def test_model_Bennu(X_Bennu, reflected, thermal, temps, time: str, validation_plots_Bennu_path):
        # Organize data to match what the ML model expects
        y_Bennu = np.zeros((len(X_Bennu[:,0]), len(C.wavelengths), 2))
        y_Bennu[:, :, 0] = reflected
        y_Bennu[:, :, 1] = thermal

        savepath = Path(validation_plots_Bennu_path, time)
        os.mkdir(savepath)

        error_dict = test_model(X_Bennu, y_Bennu, model, temps, savepath)

        return error_dict

    validation_plots_Bennu_path = Path(validation_run_folder,
                                       'bennu_validation')  # Save location of plots from validating with Bennu data
    if os.path.isdir(validation_plots_Bennu_path) == False:
        os.mkdir(validation_plots_Bennu_path)
    #
    print('Testing with Bennu data, local time 15:00')
    errors_1500 = test_model_Bennu(uncorrected_1500, corrected_1500, thermal_tail_1500, temperatures_1500, str(1500), validation_plots_Bennu_path)
    print('Testing with Bennu data, local time 12:30')
    errors_1230 = test_model_Bennu(uncorrected_1230, corrected_1230, thermal_tail_1230, temperatures_1230, str(1230), validation_plots_Bennu_path)
    print('Testing with Bennu data, local time 10:00')
    errors_1000 = test_model_Bennu(uncorrected_1000, corrected_1000, thermal_tail_1000, temperatures_1000, str(1000), validation_plots_Bennu_path)

    # Collecting calculated results into one dictionary and saving it as toml
    errors_Bennu = {}
    errors_Bennu['errors_1000'] = errors_1000
    errors_Bennu['errors_1230'] = errors_1230
    errors_Bennu['errors_1500'] = errors_1500
    FH.save_toml(errors_Bennu, Path(validation_plots_Bennu_path, 'errors_Bennu.toml'))

    # errordict = FH.load_toml(Path('validation_and_testing/validation-run_20220330-130206/bennu_validation/errors_Bennu.toml'))
    # errors_1000 = errordict['errors_1000']
    # errors_1230 = errordict['errors_1230']
    # errors_1500 = errordict['errors_1500']

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
            plt.plot(range(300, 350), range(300, 350), 'r')  # Plot a reference line with slope 1: ideal result
        plt.savefig(Path(savefolder, f'{data_name}.png'))
        # plt.show()
        plt.close(fig)


    savefolder = validation_plots_Bennu_path
    Bennuplot(errors_1000, errors_1230, errors_1500, 'predicted_temperature', 'Predicted temperature [K]', savefolder)
    Bennuplot(errors_1000, errors_1230, errors_1500, 'reflected_MAE', 'Reflected MAE', savefolder)
    Bennuplot(errors_1000, errors_1230, errors_1500, 'thermal_MAE', 'Thermal MAE', savefolder)
    Bennuplot(errors_1000, errors_1230, errors_1500, 'reflected_SAM', 'Reflected SAM', savefolder)
    Bennuplot(errors_1000, errors_1230, errors_1500, 'thermal_SAM', 'Thermal SAM', savefolder)

    # Plots where error of the uncorrected results is shown alongside the corrected
    def Bennu_comparison_plots(corrected_name, uncorrected_name, label, lim=(0,0)):
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
        # Default pyplot colors: '#1f77b4', '#ff7f0e', '#2ca02c'
        uncor_scatter1 = plt.scatter(ground_temps_1000, uncorrected_1000, alpha=0.1, color='#1f77b4')
        uncor_scatter2 = plt.scatter(ground_temps_1230, uncorrected_1230, alpha=0.1, color='#1f77b4')
        uncor_scatter3 = plt.scatter(ground_temps_1500, uncorrected_1500, alpha=0.1, color='#1f77b4')

        cor_scatter1 = plt.scatter(ground_temps_1000, corrected_1000, alpha=0.1, color='#ff7f0e')
        cor_scatter2 = plt.scatter(ground_temps_1230, corrected_1230, alpha=0.1, color='#ff7f0e')
        cor_scatter3 = plt.scatter(ground_temps_1500, corrected_1500, alpha=0.1, color='#ff7f0e')

        if lim != (0, 0):
            plt.ylim(lim)
        leg = plt.legend([cor_scatter1, uncor_scatter1], ['NN-corrected', 'Uncorrected'])  #plt.legend(('NN-corrected', 'Uncorrected'), title='Local time on Bennu')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        plt.xlabel('Ground truth temperature [K]')
        plt.ylabel(label)
        plt.savefig(Path(savefolder, f'{corrected_name}_{uncorrected_name}.png'))
        plt.close(fig)

    Bennu_comparison_plots('reflected_MAE', 'reflected_MAE_uncorrected', 'Reflected radiance MAE')#, lim=(0, 0.005))
    Bennu_comparison_plots('reflected_SAM', 'reflected_SAM_uncorrected', 'Reflected radiance cosine distance')#, lim=(0.9999, 1.0))
    Bennu_comparison_plots('reflectance_MAE', 'reflectance_MAE_uncorrected', 'Reflectance MAE')#, lim=(0, 0.01))
    Bennu_comparison_plots('reflectance_SAM', 'reflectance_SAM_uncorrected', 'Reflectance cosine distance')#, lim=(0.999, 1.0))


def validate_and_test(model):
    """
    Run validation with synthetic data and testing with real data, for a trained model given as argument.

    :param model: Keras Model -instance
        A trained model that will be tested
    """

    # Generate a unique folder name for results of test based on time the test was run
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # timestr = 'test'  # folder name for test runs, otherwise a new folder is always created

    # Create folder for results
    validation_run_folder = Path(C.val_and_test_path, f'validation-run_{timestr}')
    if os.path.isdir(validation_run_folder) == False:
        os.mkdir(validation_run_folder)

    # Print summary of model architecture into file
    with open(Path(validation_run_folder, 'modelsummary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # Validation with synthetic data similar to training data
    validate_synthetic(model, validation_run_folder)

    # Testing with real asteroid data: do not look at this until the network works properly with synthetic data
    validate_bennu(model, validation_run_folder)


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
    refl_MAE_uncor = MAE_dict['reflected_MAE_uncorrected']
    R_MAE = MAE_dict['reflectance_MAE']
    R_MAE_uncor = MAE_dict['reflectance_MAE_uncorrected']

    SAM_dict = errordict['SAM']
    therm_SAM = SAM_dict['thermal_SAM']
    refl_SAM = SAM_dict['reflected_SAM']
    refl_SAM_uncor = SAM_dict['reflected_SAM_uncorrected']
    R_SAM = SAM_dict['reflectance_SAM']
    R_SAM_uncor = SAM_dict['reflectance_SAM_uncorrected']

    def plot_and_save(data, label, filename):
        fig = plt.figure()
        plt.scatter(temperature_ground, data, alpha=0.1)
        # plt.scatter(temperature_ground, R_MAE, alpha=0.1)
        # plt.scatter(temperature_ground, R_MAE_uncor, alpha=0.1)
        plt.xlabel('Ground truth temperature')
        # plt.ylim(0,0.02)
        plt.ylabel(label)
        plt.savefig(Path(folderpath, filename))
        plt.show()
        plt.close(fig)
    # plot_and_save(np.asarray(refl_MAE_uncor) - np.asarray(refl_MAE), r'MAE($L_{th}$) improvement', 'refl-MAE-improvement_groundtemp.png')
    # plot_and_save(np.asarray(R_MAE_uncor) - np.asarray(R_MAE), r'MAE($R$) improvement', 'R-MAE-improvement_groundtemp.png')

    plot_and_save(np.asarray(refl_SAM_uncor) - np.asarray(refl_SAM), r'SAM($L_{th}$) improvement', 'refl-SAM-improvement_groundtemp.png')
    plot_and_save(np.asarray(R_SAM_uncor) - np.asarray(R_SAM), r'SAM($R$) improvement', 'R-SAM-improvement_groundtemp.png')
    plot_and_save()

    # fig = plt.figure()
    # plt.figure()
    # plt.scatter(temperature_ground, refl_SAM, alpha=0.1)
    # plt.xlabel('Ground truth temperature')
    # plt.ylabel('Reflected SAM')
    # plt.savefig(Path(folderpath, 'reflSAM_groundtemp.png'))
    # plt.close(fig)
    #
    # fig = plt.figure()
    # plt.figure()
    # plt.scatter(temperature_ground, therm_SAM, alpha=0.1)
    # plt.xlabel('Ground truth temperature')
    # plt.ylabel('Thermal SAM')
    # plt.savefig(Path(folderpath, 'thermSAM_groundtemp.png'))
    # plt.close(fig)
    #
    # fig = plt.figure()
    # plt.figure()
    # plt.scatter(temperature_ground, therm_MAE, alpha=0.1)
    # plt.xlabel('Ground truth temperature')
    # plt.ylabel('Thermal MAE')
    # plt.savefig(Path(folderpath, 'thermMAE_groundtemp.png'))
    # plt.close(fig)
    #
    # fig = plt.figure()
    # plt.figure()
    # plt.scatter(temperature_ground, refl_MAE, alpha=0.1)
    # plt.xlabel('Ground truth temperature')
    # plt.ylabel('Reflected MAE')
    # plt.savefig(Path(folderpath, 'reflMAE_groundtemp.png'))
    # plt.close(fig)


def plot_Bennu_errors(folderpath):
    errordict = file_handling.load_toml(Path(folderpath, 'errors_Bennu.toml'))
    errors_1000 = errordict['errors_1000']
    errors_1230 = errordict['errors_1230']
    errors_1500 = errordict['errors_1500']

    def Bennuplot(errors_1000, errors_1230, errors_1500, data_name, label):

        def fetch_data(errordict, data_name):
            temperature_dict = errordict['temperature']
            ground_temps = np.asarray(temperature_dict['ground_temperature'])

            if 'MAE' in data_name:
                data_dict = errordict['MAE']
                if 'reflected' in data_name:
                    data = np.asarray(data_dict['reflected_MAE'])
                else:
                    data = np.asarray(data_dict['thermal_MAE'])
            elif 'SAM' in data_name:
                data_dict = errordict['SAM']
                if 'reflected' in data_name:
                    data = np.asarray(data_dict['reflected_SAM'])
                else:
                    data = np.asarray(data_dict['thermal_SAM'])
            elif 'temperature' in data_name:
                data_dict = errordict['temperature']
                if 'predicted' in data_name:
                    data = np.asarray(data_dict['predicted_temperature'])
                elif 'error' in data_name:
                    data = ground_temps - np.asarray(data_dict['predicted_temperature'])

            return ground_temps, data

        ground_temps_1000, data_1000 = fetch_data(errors_1000, data_name)
        ground_temps_1230, data_1230 = fetch_data(errors_1230, data_name)
        ground_temps_1500, data_1500 = fetch_data(errors_1500, data_name)

        plt.figure()
        plt.scatter(ground_temps_1000, data_1000, alpha=0.1)
        plt.scatter(ground_temps_1230, data_1230, alpha=0.1)
        plt.scatter(ground_temps_1500, data_1500, alpha=0.1)
        plt.xlabel('Ground truth temperature [K]')
        plt.ylabel(label)
        leg = plt.legend(('10:00', '12:30', '15:00'), title='Local time on Bennu')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        if data_name == 'predicted_temperature':
            plt.plot(range(300, 350), range(300, 350), 'r')  # Plot a reference line with slope 1: ideal result
        plt.savefig(Path(folderpath, f'{data_name}.png'))
        # plt.show()

    # savefolder = C.val_and_test_path
    # Bennuplot(errors_1000, errors_1230, errors_1500, 'predicted_temperature', 'Predicted temperature [K]')
    # Bennuplot(errors_1000, errors_1230, errors_1500, 'reflected_MAE', 'Reflected MAE')
    # Bennuplot(errors_1000, errors_1230, errors_1500, 'thermal_MAE', 'Thermal MAE')
    # Bennuplot(errors_1000, errors_1230, errors_1500, 'reflected_SAM', 'Reflected SAM')
    # Bennuplot(errors_1000, errors_1230, errors_1500, 'thermal_SAM', 'Thermal SAM')
    print('test')

    def Bennu_comparison_plots(corrected_name, uncorrected_name, label, lim=(0,0)):
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

        # cor_scatter1 = plt.scatter(ground_temps_1000, uncorrected_1000, alpha=0.1, color='r', marker='x')#'#ff7f0e')
        # cor_scatter2 = plt.scatter(ground_temps_1230, uncorrected_1230, alpha=0.1, color='r', marker='x')#'#ff7f0e')
        # cor_scatter3 = plt.scatter(ground_temps_1500, uncorrected_1500, alpha=0.1, color='r', marker='x')#'#ff7f0e')
        # Default pyplot colors: '#1f77b4', '#ff7f0e', '#2ca02c'
        # uncor_scatter1 = plt.scatter(ground_temps_1000, np.asarray(uncorrected_1000) - np.asarray(corrected_1000), alpha=0.1)#, color='#1f77b4')
        # uncor_scatter2 = plt.scatter(ground_temps_1230, np.asarray(uncorrected_1230) - np.asarray(corrected_1230), alpha=0.1)#, color='#1f77b4')
        # uncor_scatter3 = plt.scatter(ground_temps_1500, np.asarray(uncorrected_1500) - np.asarray(corrected_1500), alpha=0.1)#, color='#1f77b4')
        uncor_scatter1 = plt.scatter(ground_temps_1000, np.asarray(uncorrected_1000) - np.asarray(corrected_1000),
                                     alpha=0.1)  # , color='#1f77b4')
        uncor_scatter2 = plt.scatter(ground_temps_1230, np.asarray(uncorrected_1230) - np.asarray(corrected_1230),
                                     alpha=0.1)  # , color='#1f77b4')
        uncor_scatter3 = plt.scatter(ground_temps_1500, np.asarray(uncorrected_1500) - np.asarray(corrected_1500),
                                     alpha=0.1)  # , color='#1f77b4')


        if lim != (0, 0):
            plt.ylim(lim)

        leg = plt.legend(('10:00', '12:30', '15:00'), title='Local time on Bennu')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        plt.xlabel('Ground truth temperature [K]')
        plt.ylabel(label)
        # plt.show()
        plt.savefig(Path(folderpath, f'{corrected_name}_improvement.png'))
        plt.close(fig)

    Bennu_comparison_plots('reflected_MAE', 'reflected_MAE_uncorrected', r'MAE($L_{refl}$) improvement)')#, lim=(-0.002, 0.004))
    Bennu_comparison_plots('reflected_SAM', 'reflected_SAM_uncorrected', r'$L_{refl}$ cosine distance improvement')#, lim=(0, 0))
    Bennu_comparison_plots('reflectance_MAE', 'reflectance_MAE_uncorrected', r'MAE($R$) improvement')#, lim=(-0.007, 0.009))
    Bennu_comparison_plots('reflectance_SAM', 'reflectance_SAM_uncorrected', r'$R$ cosine distance improvement')#, lim=(0, 0))

