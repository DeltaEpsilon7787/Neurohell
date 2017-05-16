# -*- coding: utf-8 -*-
"""
Implements some basic meta-functions for CNN.
"""

from collections import Counter, deque

import numpy as np

from keras.layers import Conv2D, Dense
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers import Flatten, Reshape, Dropout, GaussianNoise

from keras.models import Sequential, save_model, load_model
from keras.metrics import categorical_accuracy

from keras.utils import to_categorical


def generate_network(input_shape,
                     architecture,
                     architecture_data,
                     optimizer,
                     loss):
    """Generate CNN

    Inputs
    ------

    ------

    Returns
    -------
        Generated model
    -------
    """

    model = Sequential()
    char_map = dict((('C', Conv2D),
                     ('D', Dropout),
                     ('G', GaussianNoise),
                     ('A', AveragePooling2D),
                     ('M', MaxPooling2D),
                     ('F', Flatten),
                     ('R', Reshape),
                     ('L', Dense)))

    counter = Counter()
    first_layer = True
    for code in architecture:
        factory_func = char_map[code]
        data_id = code + str(counter[code])
        layer_params = architecture_data.get(data_id, {})
        layer_params['name'] = data_id

        if first_layer:
            layer_params['input_shape'] = input_shape
            first_layer = False

        new_layer = factory_func(**layer_params)
        model.add(new_layer)
        counter[code] += 1

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[categorical_accuracy])

    return model


def train_network(model,
                  data_in,
                  data_out,
                  epochs=32,
                  val_data_in=None,
                  val_data_out=None,
                  save_path=None,
                  early_stopper=None,
                  verbose=0):
    """Train given network using samples.

    Inputs
    ------
    model: Network
        Network to train
    data_in: Data 4D array (size, height, width, 1)
        Samples to be forwarded to the network
    data_out: Output 4D array (samples, 62)
        Expected output. Each row should only contain one 1
    epochs: uint, optional
        Epochs to pass before cutoff
    validation_split: float, [0..1], optional
        Ratio of training samples to test samples.
    save_path: str or None, optional
        Automatically saves trained network here if not None
    verbose: 0, 1, 2
        Supplied to fit()
    ------
    """

    model.fit(data_in,
              data_out,
              validation_data=(val_data_in, val_data_out),
              epochs=epochs,
              shuffle=True,
              callbacks=[early_stopper],
              verbose=verbose)

    if save_path:
        save_model(model, save_path)


def load_network(load_path):
    """Load network from load_path

    Inputs
    ------
    load_path: str
        Path to load network from. No validation checks are performed!
    ------

    Returns
    -------
    Loaded model
    -------
    """
    return load_model(load_path)


def save_network(model, save_path):
    """Save network to save_path

    Inputs
    ------
    save_path: str
        Creates or overwrites the file at this location
    ------
    """
    save_model(model, save_path)


def classes_to_chars(classes, mask):
    """Convert classes to chars

    Inputs
    ------
    classes: 2D uint array of shape (?, 1)
        Classes to be converted
    mask: one-to-one mapping
        Converting mapping to be applied to classes
    ------

    Returns
    -------
        2D array of chars of shape (?, 1)
    -------
    """
    return_value = deque()
    for category in classes:
        best_prediction = np.argwhere(category)
        if best_prediction.shape[0] > 1:  # More than one prediction
             return_value.append('UNKNOWN')
             continue
        return_value.append(mask[best_prediction[0][0]])
    return np.array(return_value).reshape(-1, 1)


def chars_to_classes(chars, mask, num_classes):
    return_value = deque()
    for char in chars:
        masked_out = to_categorical(mask[char], num_classes=num_classes)
        return_value.append(masked_out)
    return np.array(return_value).reshape((len(chars), num_classes))

def pass_data(model,
              data_in,
              mask,
              num_classes,
              threshold=0.7):
    """Pass data to given model. Apply mask to output.

    Inputs
    ------
    model: (preferably) trained network
        Network to pass data into
    data_in: 4D array of shape (?, height, width, 1)
        Data to pass into the network
    mask: one-to-one mapping
        Converting mapping
    num_classes: uint
        Amount of classes present
    threshold: float, [0..1], optional
        Minimum relative probability.
    ------

    Returns
    -------
        1D list of values from mask or Nones
        If multiple values are significant, then None.
        If no values are of enough significance, then None.
    -------
    """

    output = model.predict_proba(data_in)
    output = output / np.max(output, axis=1).reshape((-1, 1))
    output_mask = output < threshold
    output[output_mask] = 0.0

    return classes_to_chars(output, mask)
