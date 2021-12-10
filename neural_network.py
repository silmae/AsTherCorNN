import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from pathlib import Path
import pickle
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Concatenate
from tensorflow.keras.models import Model, load_model
import constants as C

# # For running with GPU on server:
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # Check available GPU with command nvidia-smi in terminal
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# To show plots from server, make X11 connection and add this to Run configuration > Environment variables:
# DISPLAY=localhost:10.0

def create_slope(length):
    val = (np.random.rand(1) - 0.5) * 0.3
    # val = 0.1  # Static slope for testing
    offset = np.random.rand(1)
    # offset = 0.5  # Static offset
    slope = np.linspace(0, val, length).flatten() + offset

    if np.min(slope) < 0:
        slope = slope + np.min(slope)

    return slope

def create_normal(length):
    mu = np.random.rand(1) * (length/2)
    sigma = (0.1 + np.random.rand(1)) * 0.1 * length
    # Static mu and sigma for preliminary tests:
    # mu = 20
    # sigma = 10
    normal = norm.pdf(range(length), mu, sigma) * 10

    return normal


def create_data(length, samples):

    data = np.zeros((samples, length))
    ground1 = np.zeros((samples, length))
    ground2 = np.zeros((samples, length))

    for i in range(samples):
        slope = create_slope(length)
        normal = create_normal(length) + create_normal(length)
        summed = normal + slope

        data[i, :] = summed
        ground1[i, :] = slope
        ground2[i, :] = normal

    return data, ground1, ground2


def create_model(input_length, waist_size, activation):

    # Create input for fully connected
    input_data = Input(shape=(input_length, ))

    # Calculate node count for first hidden layer by taking the nearest power of 2
    node_count = 2 ** np.floor(np.log2(input_length))
    # Create first hidden layer for encoder
    encoder = Dense(node_count, activation=activation)(input_data)

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

    # For some reason the outputs don't work with sigmoid
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
    y1 = ground[:, :, 0]
    # print(y1.shape[1])
    y2 = ground[:, :, 1]
    y1_pred = prediction[:, 0:y1.shape[1]]
    y2_pred = prediction[:, y1.shape[1]:2*y1.shape[1]+1]

    L2_dist1 = tf.norm(y1 - y1_pred, axis=1, keepdims=True)
    L2_dist2 = tf.norm(y2 - y2_pred, axis=1, keepdims=True)
    L2_dist = L2_dist1 + L2_dist2

    predict1_grad = y1_pred[:, 1:100] - y1_pred[:, 0:99]  # TODO replace the hardcoded indices IF using this gradient
    grad_norm1 = tf.norm(predict1_grad, axis=1, keepdims=True)

    predict2_grad = y2_pred[:, 1:100] - y2_pred[:, 0:99]
    grad_norm2 = tf.norm(predict2_grad, axis=1, keepdims=True)

    loss = L2_dist #+ 0.0001 * (grad_norm1 + grad_norm2)  # TODO add cos_dist?

    return loss






# data, ground1, ground2 = create_data(length, samples)
# ground = np.zeros((samples, length, 2))
# ground[:, :, 0] = ground1
# ground[:, :, 1] = ground2

def init_autoencoder(length):

    model = create_model(length, C.waist, C.activation)
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=C.learning_rate)

    # Compile model
    model.compile(optimizer=opt, loss=loss_fn)



    return model

def train_autoencoder(data, ground, early_stop=True, checkpoints=True, save_history=True):

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

    # Train model and save history
    history = model.fit([data], [ground], batch_size=C.batches, epochs=C.epochs, validation_split=0.2,
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
    filename = C.run_figname + '_history.png'
    plt.savefig(Path(C.training_path, filename))
    # plt.show()
    #

    # Return model to make predictions elsewhere
    return model

# # Testing by creating new data and predicting with the model
# for i in range(10):
#     slope = create_slope(length)
#     normal = create_normal(length) + create_normal(length)
#     summed = normal + slope
#     prediction = model.predict(np.array([summed.T])).squeeze()
#     pred1 = prediction[0:int(len(prediction)/2)]
#     pred2 = prediction[int(len(prediction)/2):len(prediction) + 1]
#
#     plt.figure()
#     x = range(length)
#     plt.plot(x, slope, 'r')
#     plt.plot(x, normal, 'b')
#     plt.plot(x, pred1.squeeze(), '--c')
#     plt.plot(x, pred2.squeeze(), '--m')
#
#     plt.legend(('ground 1', 'ground 2', 'prediction 1', 'prediction 2'))
#
# plt.show(block=True)



