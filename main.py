
import os

from matplotlib import pyplot as plt

# PyPlot settings to be used in all plots
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'savefig.dpi': 600})
# plt.rcParams['text.usetex'] = True

import utils
import constants as C
import reflectance_data as refl
import radiance_data as rad
import file_handling as FH
import neural_network as NN

if __name__ == '__main__':
    ############################
    # TEST SANDBOX

    print('test')
    ############################
    # For running with GPU on server (having these lines here won't hurt when running locally without GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Check available GPU with command nvidia-smi in terminal, pick one that is not in use
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    ############################
    # # HYPERPARAMETER OPTIMIZATION
    # NN.tune_model(300, 20, 1)

    ############################
    # # TRAINING
    # # NN.prepare_training_data()
    # # Create a neural network model
    # untrained = NN.create_model(
    #     conv_filters=C.conv_filters,
    #     conv_kernel=C.conv_kernel,
    #     encoder_start=C.encoder_start,
    #     encoder_node_relation=C.encoder_node_relation,
    #     encoder_stop=C.encoder_stop,
    #     lr=C.learning_rate
    # )
    #
    # # # Load weights to continue training where you left off:
    # # last_epoch = 445
    # # weight_path = Path(C.weights_path, f'weights_{str(last_epoch)}.hdf5')
    # # untrained.load_weights(weight_path)
    #
    # # Train the model
    # model = NN.train_network(untrained, early_stop=False, checkpoints=True, save_history=True, create_new_data=False)

    ##############################
    # VALIDATION
    import validation as val  # TODO This uses symfit, which I have not installed on my thingfish conda env

    # Run validation with synthetic data and test with real data
    last_epoch = 1372
    val.validate_and_test(last_epoch)

    ############################
    # TODO Poistaakko?
    # # MORE VALIDATION
    # val.error_plots(Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/validation_and_testing/validation-run_20220330-162708/synthetic_validation'))
    # val.plot_Bennu_errors('//home/leevi/PycharmProjects/asteroid-thermal-modeling/validation_and_testing/validation-run_20220421-103518/bennu_validation')

    # # Loading errors from Bennu testing, plotting results
    # errordict = FH.load_toml(Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/figs/validation_plots/validation-run_20220301-135222/bennu_validation/errors_Bennu.toml'))
    # val.plot_Bennu_errors(errordict)
    #
    # Plotting some errors as function of ground truth temperature
    # folderpath = Path('/home/leevi/PycharmProjects/asteroid-thermal-modeling/figs/validation_plots/validation-run_20220224-110013/synthetic_validation')
    # val.error_plots(folderpath)

    ##############################








