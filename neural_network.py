"""
Methods for building, tuning, and using 1D convolutional neural networks
"""

import os
import time
import csv
from pathlib import Path
from contextlib import redirect_stdout  # For saving keras prints into text files
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D
from tensorflow.keras.models import Model, load_model
import keras_tuner as kt
from tensorflow.keras.callbacks import CSVLogger
import sklearn.utils

import constants as C
import reflectance_data as refl
import radiance_data as rad
import file_handling as FH


def prepare_training_data():
    """
    Creating training and validation data. Takes reflectance spectra of asteroids, and divides them into a larger set
    for training and a smaller set for validation. From each reflectance creates a number of simulated radiances with
    random values for heliocentric distance, incidence and emission angles, and surface temperature.

    For training the sum of reflected and thermal radiances is the input, and temperature is the ground truth. Function
    creates dictionaries for all simulated observations, writing into them the three spectra, and metadata
    related to parameters used in their creation. Each dictionary is saved into its own .toml file.
    """

    '''
    Meteorite reflectances (Gaffey, for example) could be used for simulating reflected radiances, but they are likely 
    not very good stand-ins for asteroid reflectance: no space weathering, and possible changes from atmospheric shock.
    They could possibly simulate fresh regolith in a future version.
    '''

    # Load asteroid reflectances
    train_reflectances, test_reflectances = refl.read_asteroids()

    # Calculate a number of radiances from each reflectance, save them on disc as toml, and return the radiances
    # and their temperature and emissivity
    radiances_test, parameters_test = rad.calculate_radiances(test_reflectances, test=True, samples_per_temperature=int(len(test_reflectances)/5), emissivity_type='random')
    radiances_training, parameters_training = rad.calculate_radiances(train_reflectances, test=False, samples_per_temperature=int(len(train_reflectances)/5), emissivity_type='random')

    # Create a "bunch" from training and testing radiances and save both in their own files. This is orders of
    # magnitude faster than reading each radiance from its own toml
    def bunch_rads(radiances, parameters, filepath: Path):
        rad_bunch = {}
        rad_bunch['radiances'] = radiances
        rad_bunch['parameters'] = parameters

        FH.save_pickle(rad_bunch, filepath)

    bunch_rads(radiances_test, parameters_test, C.rad_bunch_test_path)
    bunch_rads(radiances_training, parameters_training, C.rad_bunch_training_path)


def tune_model(epochs: int, max_trials: int, executions_per_trial: int):
    """
    Tune the model architecture using KerasTuner

    :param epochs: int
        Number of epochs trained for every trial
    :param max_trials: int
        Maximum number of trial configurations to be tested
    :param executions_per_trial: int
        How many times each tested configuration is trained (values larger than one used to reduce the effects of bad
        random values)
    """

    # Hyperparameter optimization with KerasTuner's Bayesian optimizer
    savefolder_name = f'optimization-run_{time.strftime("%Y%m%d-%H%M%S")}'
    tuner = kt.BayesianOptimization(
        hypermodel=create_hypermodel,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        overwrite=True,
        directory=C.hyperparameter_path,
        project_name=savefolder_name,
    )

    tuner.search_space_summary()
    x_train, y_train, x_val, y_val = load_training_validation_data()

    tuner.search(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))

    tuner.results_summary()

    # Save summary of results into a text file
    tuning_results_path = Path(C.hyperparameter_path, savefolder_name)
    with open(Path(tuning_results_path, 'trial_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            tuner.results_summary()


def create_hypermodel(hp):
    """
    Create and compile a neural network model for hyperparameter optimization. Model structure is similar to unadjustable
    network: dense input, conv1d, dense autoencoder, conv1d, dense output, concatenate. This method calls the
    create_model -method using the hyperparameters as arguments.

    Adjustable hyperparameters are:
    - convolution filter count (of first conv layer, reduces in subsequent) and kernel width (same for all conv layers),
    - encoder start layer node count, relation of subsequent autoencoder layer node counts, waist layer node count,
    - learning rate

    :param hp:
    Instance of KerasTuner's kt.HyperParameters()

    :return: model
    Compiled Keras Model -instance
    """

    # Convolution layer for the input, tune both filter number and kernel size
    filters = hp.Int("filters", min_value=10, max_value=80, step=10)
    kernel_size = hp.Int("kernel_size", min_value=5, max_value=60, step=5)

    # Tune encoder start layer node count
    encoder_start = hp.Int('encdec_start', min_value=200, max_value=1200, step=50)
    # Tune number of nodes in last encoder layer
    encoder_end = hp.Int('waist_size', min_value=20, max_value=300, step=10)
    # Tune the relation between node counts of subsequent encoder layers: (layer N nodes) / (layer N-1 nodes)
    encoder_node_relation = hp.Float("encdec_node_relation", min_value=0.1, max_value=0.9, sampling="linear")

    # Tune learning rate of the model
    lr = hp.Float('lr', min_value=1e-8, max_value=1e-4, sampling='log')

    # Create model in separate function with adjustable hyperparameters as inputs
    model = create_model(filters, kernel_size, encoder_start, encoder_node_relation, encoder_end, lr)

    return model


def create_model(conv_filters: int, conv_kernel: int, encoder_start: int, encoder_node_relation: float, encoder_stop: int, lr: float):
    """
    Create and compile a dense encoder model with 1D convolution at start, using Keras.

    Network structure:
    dense input --> conv1d layers --> dense encoder --> dense output

    :param conv_filters: int
        Number of filters in convolutional layer
    :param conv_kernel: int
        Kernel width in the convolution
    :param encoder_start: int
        Node count at the start of the encoder
    :param encoder_node_relation:
        Relation between node counts of subsequent encoder layers: (layer N nodes) / (layer N-1 nodes)
    :param encoder_stop: int
        Node count of last autoencoder layer before output
    :param lr: float
        Learning rate in training the network

    :return: Keras Model -instance
        Compiled model ready for training
    """

    # Input layer always depends on the input data, which depends on the wl-vector specified in constants
    input_length = len(C.wavelengths)
    input_data = Input(shape=(input_length, 1))

    # 1D convolution layer(s)
    conv1 = Conv1D(filters=conv_filters, kernel_size=conv_kernel, padding='same', activation=C.activation, strides=1)(input_data)
    conv1 = Conv1D(filters=int(conv_filters/2), kernel_size=conv_kernel, padding='same', activation=C.activation, strides=1)(conv1)
    conv1 = Conv1D(filters=int(conv_filters/4), kernel_size=conv_kernel, padding='same', activation=C.activation, strides=1)(conv1)
    conv1 = Conv1D(filters=int(conv_filters/8), kernel_size=conv_kernel, padding='same', activation=C.activation, strides=1)(conv1)

    # Flatted to make conv output compatible with following dense layer
    conv1 = Flatten()(conv1)

    # Create dense encoder based on start, relation, and end. See docstring of this method for description of node
    # relation parameter
    node_count = encoder_start
    encoder = Dense(node_count, activation=C.activation)(conv1)
    counts = [node_count]  # Save node counts of all layers into a list
    while node_count * encoder_node_relation >= encoder_stop:
        node_count = int(node_count * encoder_node_relation)
        encoder = Dense(node_count, activation=C.activation)(encoder)
        counts.append(node_count)

    # Output with one neuron, the predicted temperature
    output = Dense(1, activation='linear')(encoder)

    # Create a model object and print summary
    model = Model(inputs=[input_data], outputs=[output])
    model.summary()

    # Define optimizer, set learning rate of the model
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compile model
    model.compile(optimizer=opt, loss=tf.keras.losses.MeanAbsolutePercentageError())  # Native Keras MAPE as loss
    # model.compile(optimizer=opt, loss=loss_fn)  # Using a home-made loss function

    return model


def loss_fn(ground, prediction):
    """
    Calculate loss from predicted value and corresponding ground truth.

    :param ground: tf.Tensor
        Ground truth, temperature value for every item in batch
    :param prediction: tf.Tensor
        Prediction, temperature value for every item in batch

    :return:
        Calculated loss
    """

    # Calculate loss from ground truth and prediction
    loss = tf.math.abs(ground - prediction)

    # Printing loss into console (since debugger will not show tensor values)
    # tf.compat.v1.control_dependencies([tf.print(loss)])

    return loss


def load_training_validation_data():
    """
    Load training and validation data and ground truths from files and return them: training data from one pickle file,
    validation data from another.

    :return: x_train, y_train, x_val, y_val
    Training data, training ground truth, validation data, validation ground truth
    """

    # Load training radiances from one file as dicts
    rad_bunch_training = FH.load_pickle(C.rad_bunch_training_path)
    x_train = rad_bunch_training['radiances']
    y_train = rad_bunch_training['parameters']
    y_train = y_train[:, 0]  # Both temperature and emissivity are saved here, pick only temperature

    # Shuffle the data, otherwise the network will learn its order
    x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=0)

    # Load validation radiances from one file as dicts
    rad_bunch_test = FH.load_pickle(C.rad_bunch_test_path)
    x_val = rad_bunch_test['radiances']
    y_val = rad_bunch_test['parameters']
    y_val = y_val[:, 0]  # Both temperature and emissivity are saved here, pick only temperature
    x_val, y_val = sklearn.utils.shuffle(x_val, y_val, random_state=0)

    return x_train, y_train, x_val, y_val


def train_network(model, early_stop=True, checkpoints=True, save_history=True, create_new_data=False, logging=True):
    """
    Train deep learning model according to arguments and other parameters set in constants.py.

    :param model:
        Compiled Keras Model that will be trained
    :param early_stop:
        Whether early stop will be used or not. Patience and minimum change are set in constants.py
    :param checkpoints:
        Whether checkpoint weights are saved every time val. loss improves
    :param save_history:
        Whether loss history will be saved in a file after training is complete
    :param create_new_data:
        Whether new data will be created or old data will be loaded from disc
    :param logging:
        Whether training loss outputs will be written into a csv file during training

    :return:
        Trained model
    """

    # Early stop callback
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=C.min_delta,
        patience=C.patience,
        mode='auto',
        restore_best_weights=True
    )

    # Save the weights callback, only saves when validation loss has improved
    extract_destination = C.weights_path
    checkpoint_filepath = os.path.join(extract_destination, 'weights_{epoch}.hdf5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
        save_freq='epoch')

    # Logging callback, writes the training prints into a file and saves it
    log_path = Path(C.training_run_path, 'training.log')
    csv_logger = CSVLogger(log_path, append=True)

    # List of callbacks, append the ones to be used
    model_callbacks = []
    if early_stop == True:
        model_callbacks.append(early_stop_callback)
    if checkpoints == True:
        model_callbacks.append(model_checkpoint_callback)
    if logging == True:
        model_callbacks.append(csv_logger)

    # Create training and validation data from scratch, if specified in the arguments
    if create_new_data == True:
        prepare_training_data()

    # Load training and validation data from disc
    x_train, y_train, x_val, y_val = load_training_validation_data()

    # Train model
    history = model.fit([x_train], [y_train], batch_size=C.batch_size, epochs=C.epochs, validation_data=(x_val, y_val),
                        callbacks=model_callbacks)

    # Save training history
    if save_history == True:
        FH.save_pickle(history.history, C.training_history_path)

    # Summarize and plot history of loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.yscale('log')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    filename = C.training_run_name + '_history.png'
    plt.savefig(Path(C.training_run_path, filename), dpi=600)

    # Return model to make predictions elsewhere
    return model


def plot_loss_history(filter_width=15):
    """
    Plot training and validation loss history. Load log file where the histories are stored, filter the noisy
    signal with a median filter before plotting.

    :param filter_width:
        Width of median filter, accepts only odd integer values
    """

    # Plotting the loss history of the training to find out when overfitting started
    # Load training log from file
    logpath = Path(C.training_run_path, 'training.log')
    with open(logpath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        for row in reader:
            data.append(row)  # Take first (and only) element in row, convert to int and append to list

    data = np.asarray(data)
    epoch = data[1:, 0].astype(int)
    train_loss = data[1:, 1].astype(float)
    val_loss = data[1:, 2].astype(float)

    # Use a median filter to make sense out of the noisy loss history
    val_loss = medfilt(val_loss, filter_width)

    # Plot filtered loss histories to find out when overfitting starts
    plt.figure()
    plt.plot(epoch, train_loss)
    plt.plot(epoch, val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.show()