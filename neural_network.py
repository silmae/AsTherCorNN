"""
Methods for building and using neural networks for separating reflected and thermally emitted radiances.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import time
from pathlib import Path
import pickle
from contextlib import redirect_stdout  # For saving keras prints into text files
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Concatenate, Reshape
from tensorflow.keras.models import Model, load_model
import keras_tuner as kt

import constants as C
import reflectance_data as refl
import radiance_data as rad
import file_handling as FH


def prepare_training_data():
    """
    Creating training and validation data. Takes reflectance spectra of asteroids, and divides them into a larger set
    for training and a smaller set for validation. From each reflectance creates 10 simulated radiances with random
    values for heliocentric distance, incidence and emission angles, and surface temperature. For training the
    separate reflected and thermal radiances are the ground truth, and the sum of these is the input. Function
    creates dictionaries for all simulated observations, writing into them the three spectra, and metadata
    related to parameters used in their creation. Each dictionary is saved into its own .toml file.


    """
    # #############################  # TODO Take meteorite reflectances also into account?
    # # Load meteorite reflectances from files and create more from them through augmentation
    # train_reflectances, test_reflectances = refl.read_meteorites(waves)
    # refl.augmented_reflectances(train_reflectances, waves, test=False)
    # refl.augmented_reflectances(test_reflectances, waves, test=True)
    # #############################

    #############################
    # Load asteroid reflectances, they are already augmented
    train_reflectances, test_reflectances = refl.read_asteroids()

    # Calculate 10 radiances from each reflectance, save them on disc as toml, and return the data vectors
    summed_test, separate_test = rad.calculate_radiances(test_reflectances, test=True)
    summed_training, separate_training = rad.calculate_radiances(train_reflectances, test=False)

    # Create a "bunch" from training and testing radiances and save both in their own files. This is orders of
    # magnitude faster than reading each radiance from its own toml
    def bunch_rads(summed, separate, filepath: Path):
        rad_bunch = {}
        rad_bunch['summed'] = summed
        rad_bunch['separate'] = separate

        FH.save_pickle(rad_bunch, filepath)

    # summed_test, separate_test = rad.read_radiances(test=True)
    bunch_rads(summed_test, separate_test, C.rad_bunch_test_path)
    # summed_training, separate_training = rad.read_radiances(test=False)
    bunch_rads(summed_training, separate_training, C.rad_bunch_training_path)


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

    # Hyperparameter optimization with KerasTuner
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
    Create and compile a neural network model for hyperparameter optimization. Structure is similar to unadjustable
    network: dense input, conv1d, dense autoencoder, conv1d, dense output, concatenate. This function calls the
    create_model -function using the hyperparameters as arguments.

    Adjustable hyperparameters are:
    - convolution filter count and kernel width,
    - encoder start layer node count, relation of subsequent autoencoder layer node counts, waist layer node count,
    - learning rate

    :param hp:
    Instance of KerasTuner's kt.HyperParameters()

    :return: model
    Compiled Keras Model -instance
    """

    # Convolution layer for the input, tune both filter number and kernel size
    filters = hp.Int("filters", min_value=20, max_value=60, step=4)
    kernel_size = hp.Int("kernel_size", min_value=20, max_value=40, step=2)

    # Tune encoder/decoder start layer node count
    encdec_start = hp.Int('encdec_start', min_value=800, max_value=1200, step=40)
    # Tune number of nodes in waist layer
    waist_size = hp.Int('waist_size', min_value=80, max_value=160, step=8)
    # Tune the relation between node counts of subsequent encoder layers: (layer N nodes) / (layer N-1 nodes)
    encdec_node_relation = hp.Float("encdec_node_relation", min_value=0.1, max_value=0.5, sampling="linear")

    # Define optimizer, tune learning rate of the model
    lr = hp.Float('lr', min_value=1e-7, max_value=1e-5, sampling='log')

    # Create model in separate function with adjustable hyperparameters as inputs
    model = create_model(filters, kernel_size, encdec_start, encdec_node_relation, waist_size, lr)

    return model


def create_model(conv_filters: int, conv_kernel: int, encdec_start: int, encdec_node_relation: float, waist_size: int, lr: float):
    """
    Create and compile an autoencoder using Keras.

    Network structure:
    dense input --> conv1d --> dense autoencoder --> conv1d --> dense output --> concatenate

    :param conv_filters: int
        Number of filters in each convolutional layer
    :param conv_kernel: int
        Kernel width in the convolution
    :param encdec_start: int
        Node count at the start of the encoder / at the end of the decoder
    :param encdec_node_relation:
        Relation between node counts of subsequent encoder layers: (layer N nodes) / (layer N-1 nodes)
    :param waist_size: int
        Node count of autoencoder waist layer
    :param lr: float
        Learning rate in training the network

    :return: Keras Model -instance
        Compiled model ready for training
    """

    # Input layer always depends on the input data, which depends on the wl-vector specified in constants
    input_length = len(C.wavelengths)
    input_data = Input(shape=(input_length, 1))

    # Convolution layer
    conv1 = Conv1D(filters=conv_filters, kernel_size=conv_kernel, padding='same', activation=C.activation)(input_data)
    conv1 = Flatten()(conv1)

    # Create encoder based on start, relation, and waist
    node_count = encdec_start
    encoder = Dense(node_count, activation=C.activation)(conv1)
    counts = [node_count]  # Save node counts of all layers into a list
    while node_count * encdec_node_relation > waist_size:
        node_count = int(node_count * encdec_node_relation)
        encoder = Dense(node_count, activation=C.activation)(encoder)
        counts.append(node_count)

    waist = Dense(units=waist_size, activation=C.activation)(encoder)

    # Create a decoder identical to the encoder
    i = 0
    for node_count in reversed(counts):
        if i == 0:
            decoder = Dense(node_count, activation=C.activation)(waist)
            i = i + 1
        else:
            decoder = Dense(node_count, activation=C.activation)(decoder)

    # Convolutional layer to match the encoder side
    decoder = Reshape(target_shape=(int(node_count), 1))(decoder)
    decoder = Conv1D(filters=conv_filters, kernel_size=conv_kernel, padding='same', activation=C.activation)(decoder)
    decoder = Flatten()(decoder)

    # Output with shape similar to input
    output = Dense(input_length, activation='linear')(decoder)

    # Create a model object
    model = Model(inputs=[input_data], outputs=[output])
    model.summary()

    # Define optimizer, set learning rate of the model
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compile model
    model.compile(optimizer=opt, loss=loss_fn)

    return model


def loss_fn(ground, prediction):
    """
    Calculate loss from predicted thermal radiance spectrum and corresponding ground truth.

    :param ground: tf.Tensor
        Ground truth, two vectors for every item in batch (reflected and thermal)
    :param prediction: tf.Tensor
        Prediction, one vector for every item in batch: thermal radiance

    :return:
        Calculated loss
    """

    # Take only the thermal vector from ground truth
    ground = ground[:, :, 1]

    # Scaling the ground and prediction values: if not scaled, higher radiances will dominate, and lower will not
    # be seen as errors. Should not affect network output units when done inside loss function
    ground_max = tf.math.reduce_max(ground)
    # To prevent division by (near) zero, add small constant value to maxima
    ground_max = ground_max + 0.00001

    # Scale both ground truth and predictions by dividing with maximum
    y1 = tf.math.divide(ground, ground_max)
    y1_pred = tf.math.divide(prediction, ground_max)

    L2_dist = tf.norm(ground - prediction, axis=1, keepdims=True)

    # Cosine distance: only thermal, since those are always similar to each other in shape
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    cos_dist = cosine_loss(ground, prediction) + 1  # According to Keras documentation, -1 means similar and 1 means dissimilar: add 1 to stay positive!

    # Calculate loss as sum of L2 distance and cos distance
    loss = L2_dist + cos_dist

    # # Printing loss into console (since debugger will not show tensor values)
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
    x_train = rad_bunch_training['summed']
    y_train = rad_bunch_training['separate']

    # Load validation radiances from one file as dicts
    rad_bunch_test = FH.load_pickle(C.rad_bunch_test_path)
    x_val = rad_bunch_test['summed']
    y_val = rad_bunch_test['separate']

    return x_train, y_train, x_val, y_val


def train_autoencoder(model, early_stop=True, checkpoints=True, save_history=True, create_new_data=False):

    # Early stop callback
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=C.min_delta,
        patience=C.patience,
        mode='auto',
        restore_best_weights=True
    )

    # Save the weights callback
    extract_destination = C.weights_path
    checkpoint_filepath = os.path.join(extract_destination, 'weights_{epoch}.hdf5')
    # parameters for saving a model everytime we finish training for 1 epoch
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
        save_freq='epoch')

    # List of callbacks, append the ones to be used
    model_callbacks = []
    if early_stop == True:
        model_callbacks.append(early_stop_callback)
    if checkpoints == True:
        model_callbacks.append(model_checkpoint_callback)

    # Create training data from scratch, if specified in the arguments
    if create_new_data == True:
        prepare_training_data()

    data, ground, X_val, y_val = load_training_validation_data()

    # Train model and save history
    history = model.fit([data], [ground], batch_size=C.batches, epochs=C.epochs, validation_data=(X_val, y_val),
                        callbacks=model_callbacks)

    # Save training history
    if save_history == True:
        FH.save_pickle(history.history, C.training_history_path)

    # Summarize and plot history of loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.yscale('log')
    plt.title('Model loss history')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    filename = C.training_run_name + '_history.png'
    plt.savefig(Path(C.training_run_path, filename), dpi=600)

    # Return model to make predictions elsewhere
    return model
