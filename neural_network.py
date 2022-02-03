import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from pathlib import Path
import pickle

# from tensorboard.errors import InvalidArgumentError
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Concatenate, Reshape
from tensorflow.keras.models import Model, load_model

import constants as C
import reflectance_data as refl
import radiance_data as rad

# # For running with GPU on server:
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # Check available GPU with command nvidia-smi in terminal
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# To show plots from server, make X11 connection and add this to Run configuration > Environment variables:
# DISPLAY=localhost:10.0

def prepare_training_data():
    # #############################
    # # Load meteorite reflectances from files and create more from them through augmentation
    # train_reflectances, test_reflectances = refl.read_meteorites(waves)
    # refl.augmented_reflectances(train_reflectances, waves, test=False)
    # refl.augmented_reflectances(test_reflectances, waves, test=True)
    # #############################

    #############################
    # Load asteroid reflectances, they are already augmented
    train_reflectances, test_reflectances = refl.read_asteroids()

    ############################
    # Calculate 10 radiances from each reflectance, and save them on disc as toml
    rad.calculate_radiances(test_reflectances, test=True)
    rad.calculate_radiances(train_reflectances, test=False)

    # ##############################
    # Create a "bunch" from training and testing radiances and save both in their own files. This is orders of
    # magnitude faster than reading each radiance from its own toml

    def bunch_rads(summed, separate, filepath: Path):
        rad_bunch = {}
        rad_bunch['summed'] = summed
        rad_bunch['separate'] = separate

        with open(filepath, 'wb') as file_pi:
            pickle.dump(rad_bunch, file_pi)

    summed_test, separate_test = rad.read_radiances(test=True)
    bunch_rads(summed_test, separate_test, C.rad_bunch_test_path)

    summed_training, separate_training = rad.read_radiances(test=False)
    bunch_rads(summed_training, separate_training, C.rad_bunch_training_path)


def create_model(input_length, waist_size, activation):

    # Create input for fully connected
    # input_data = Input(shape=(input_length))

    # Input if using convolutional layer
    input_data = Input(shape=(input_length, 1))

    # Convolution layer, mainly for noise reduction
    conv1 = Conv1D(filters=16, kernel_size=3, padding='same', activation=C.activation)(input_data)
    conv1 = Flatten()(conv1)

    # Calculate node count for first autoencoder layer by taking the nearest power of 2
    node_count = 2 ** np.floor(np.log2(input_length))
    # Create first hidden layer for encoder
    encoder = Dense(node_count, activation=activation)(conv1)

    counts = [node_count]
    while node_count/2 > waist_size:
        node_count = node_count / 2
        encoder = Dense(node_count, activation=activation)(encoder)
        counts.append(node_count)
        # print(node_count)

    # Create waist layer
    waist = Dense(waist_size, activation=activation)(encoder)

    # Create two decoders, one for each output
    i = 0
    for node_count in reversed(counts):
        if i == 0:
            decoder1 = Dense(node_count, activation=activation)(waist)
            decoder2 = Dense(node_count, activation=activation)(waist)
            i = i + 1
        else:
            decoder1 = Dense(node_count, activation=activation)(decoder1)
            decoder2 = Dense(node_count, activation=activation)(decoder2)

    # Convolutional layer to match the encoder side
    decoder1 = Reshape(target_shape=(int(node_count), 1))(decoder1)
    decoder1 = Conv1D(filters=16, kernel_size=3, padding='same', activation=C.activation)(decoder1)
    decoder1 = Flatten()(decoder1)
    decoder2 = Reshape(target_shape=(int(node_count), 1))(decoder2)
    decoder2 = Conv1D(filters=16, kernel_size=3, padding='same', activation=C.activation)(decoder2)
    decoder2 = Flatten()(decoder2)

    output1 = Dense(input_length, activation='linear')(decoder1)
    output2 = Dense(input_length, activation='linear')(decoder2)

    # Concatenate the two outputs into one vector to transport it to loss fn
    conc = Concatenate()([output1, output2])

    # Create a model object(?) and return it
    model = Model(inputs=[input_data], outputs=[conc])

    return model


def loss_fn(ground, prediction):
    # print(tf.shape(ground))
    # print(tf.shape(prediction))
    #TODO Adding 1 to each element to escape problems?
    y1 = ground[:, :, 0]
    y2 = ground[:, :, 1]
    y1_pred = prediction[:, 0:y1.shape[1]]
    y2_pred = prediction[:, y1.shape[1]:]

    # Scaling the ground and prediction values: if not scaled, higher radiances will dominate, and lower will not
    # be seen as errors. Should not affect network output units when done inside loss function
    y1_max = tf.math.reduce_max(y1)
    y2_max = tf.math.reduce_max(y2)
    # To prevent division by (near) zero, add small constant value to maxima
    y1_max = y1_max + 0.00001
    y2_max = y2_max + 0.00001
    # y1_max = y1_max + tf.cond(tf.math.less_equal(y1_max, 0.00001), lambda: 0.00001, lambda: 0.0)
    # y2_max = y2_max + tf.cond(tf.math.less_equal(y2_max, 0.00001), lambda: 0.00001, lambda: 0.0)
    # Scale both ground truth and predictions by dividing with maximum
    y1 = tf.math.divide(y1, y1_max)
    y2 = tf.math.divide(y2, y2_max)
    y1_pred = tf.math.divide(y1_pred, y1_max)
    y2_pred = tf.math.divide(y2_pred, y2_max)
    # tf.compat.v1.control_dependencies([tf.print(y1_max)])  # This will print the loss into console

    L2_dist1 = tf.norm(y1 - y1_pred, axis=1, keepdims=True)
    # L2_dist1 = 0  # Leaving reflected out of loss calculation, to see if performance with therm is better
    L2_dist2 = tf.norm(y2 - y2_pred, axis=1, keepdims=True)
    L2_dist = L2_dist1 + L2_dist2

    # Cosine distance: only thermal, since those are always similar to each other
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    # cos1 = cosine_loss(y1, y1_pred)
    cos2 = cosine_loss(y2, y2_pred) + 1  # According to Keras documentation, -1 means similar and 1 means dissimilar: add 1 to stay positive!


    # cos_sum = cos1 + cos2 + 2  # According to Keras documentation, -1 means similar and 1 means dissimilar: add 2 to stay positive!

    # # Calculate minimum values of predictions
    # y1_pred_min = tf.math.reduce_min(y1_pred)
    # y2_pred_min = tf.math.reduce_min(y2_pred)
    # mincost = 0.

    # If the minimum of prediction is less than zero, add punishment to the loss
    # mincost = mincost + tf.cond(tf.math.less_equal(y1_pred_min, 0.), lambda: 1000000.0, lambda: 0.0)
    # mincost = mincost + tf.cond(tf.math.less_equal(y2_pred_min, 0.), lambda: tf.math.abs(y2_pred_min) * C.loss_negative_penalty_multiplier, lambda: 0.0)

    # predict1_grad = y1_pred[:, 1:] - y1_pred[:, 0:-1]
    # grad_norm1 = tf.norm(predict1_grad, axis=1, keepdims=True)
    #
    # predict2_grad = y2_pred[:, 1:] - y2_pred[:, 0:-1]
    # grad_norm2 = tf.norm(predict2_grad, axis=1, keepdims=True)

    loss = L2_dist + cos2 #+ (C.loss_gradient_multiplier * (grad_norm1 + grad_norm2)) #+ mincost  # TODO add cos_dist?

    # tf.compat.v1.control_dependencies([tf.print(loss)])  # This will print the loss into console

    return loss


def init_autoencoder(length):

    model = create_model(length, C.waist, C.activation)
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=C.learning_rate)

    # Compile model
    model.compile(optimizer=opt, loss=loss_fn)

    return model


def train_autoencoder(early_stop=True, checkpoints=True, save_history=True, create_new_data=False):

    # Initialize autoencoder architecture based on the number of wavelength channels
    length = len(C.wavelengths)
    model = init_autoencoder(length)

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

    # Load training radiances from one file as dicts
    with open(C.rad_bunch_training_path, 'rb') as file_pi:
        rad_bunch_training = pickle.load(file_pi)
    data = rad_bunch_training['summed']
    ground = rad_bunch_training['separate']

    # Load validation radiances from one file as dicts
    with open(C.rad_bunch_test_path, 'rb') as file_pi:
        rad_bunch_test = pickle.load(file_pi)

    X_val = rad_bunch_test['summed']
    y_val = rad_bunch_test['separate']

    # Train model and save history
    history = model.fit([data], [ground], batch_size=C.batches, epochs=C.epochs, validation_data=(X_val, y_val),
                        callbacks=model_callbacks)

    # Save training history
    if save_history == True:
        with open(C.training_history_path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

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
    plt.savefig(Path(C.training_run_path, filename), dpi=300)

    # Return model to make predictions elsewhere
    return model


def load_model(weight_path):
    model = init_autoencoder(len(C.wavelengths))
    model.load_weights(weight_path)

    return model


# # Toy problem data creation
# def create_slope(length):
#     val = (np.random.rand(1) - 0.5) * 0.3
#     # val = 0.1  # Static slope for testing
#     offset = np.random.rand(1)
#     # offset = 0.5  # Static offset
#     slope = np.linspace(0, val, length).flatten() + offset
#
#     if np.min(slope) < 0:
#         slope = slope + np.min(slope)
#
#     return slope
#
# def create_normal(length):
#     mu = np.random.rand(1) * (length/2)
#     sigma = (0.1 + np.random.rand(1)) * 0.1 * length
#     # Static mu and sigma for preliminary tests:
#     # mu = 20
#     # sigma = 10
#     normal = norm.pdf(range(length), mu, sigma) * 10
#
#     return normal
#
#
# def create_data(length, samples):
#
#     data = np.zeros((samples, length))
#     ground1 = np.zeros((samples, length))
#     ground2 = np.zeros((samples, length))
#
#     for i in range(samples):
#         slope = create_slope(length)
#         normal = create_normal(length) + create_normal(length)
#         summed = normal + slope
#
#         data[i, :] = summed
#         ground1[i, :] = slope
#         ground2[i, :] = normal
#
#     return data, ground1, ground2