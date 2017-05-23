# -*- coding: utf-8 -*-
"""
Implements some basic meta-functions used by CNN.
"""

import os
from collections import Counter, deque
from itertools import repeat
from random import choice, random, uniform
from preprocessor import converters

import numpy as np
import claptcha

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
    """Generate a CNN

    Parameters
    ----------
    input_shape: 3-tuple
        Shape of the input for this CNN
    architecture: str
        Symbolic architecture of the CNN
    architecture_data: Dict
        Configuration of layers in the CNN
    optimizer: str
        Optimizer used by CNN
    loss: str
        Loss function used by CNN
    ---------

    Returns
    -------
    KerasModel
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
        layer_params['name'] = layer_params.get('name', data_id)

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


def training_data_generator(**kwargs):
    """Return an infinite generator for characters

    Parameters
    ----------
    characters: str, default 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
        Characters that this generator can generate.
    inject_line_probability: float, default 0.5
        Probability of drawing a line in an image
    morphological_coefficient_min: float, default 0.2
    morphological_coefficient_max: float, default 0.7
        Bounds for morph. coefficient
    noise_coefficients_min: float, default 0.0
    noise_coefficients_max: float, default 1.0
        Bounds for noise coefficient.
    samples_per_batch: uint, default 50
        Amount of samples in a batch
    font_directory: str, default ./Fonts
        Path to directory with fonts
    ----------

    Returns
    -------
    Generator object
    -------
    """

    samples = deque()
    answers = deque()

    characters = kwargs.get('characters', ascii_uppercase+digits)
    line_probability = kwargs.get('inject_line_probability', 0.5)
    morph_min = kwargs.get('morphological_coefficient_min', 0.2)
    morph_max = kwargs.get('morphological_coefficient_max', 0.7)
    noise_min = kwargs.get('noise_coefficient_min', 0.0)
    noise_max = kwargs.get('noise_coefficient_max', 1.0)
    samples_per_batch = kwargs.get('samples_per_batch', 50)
    font_directory = kwargs.get('font_directory', './Fonts')

    for character in repeat(characters):
        for sample_id in range(samples_per_batch):
            random_font = choice(os.listdir(font_directory))
            injectLine = random() < line_probability
            test_captcha = claptcha.Claptcha(character,
                                             random_font,
                                             injectLine=injectLine,
                                             morph_min=morph_min,
                                             morph_max=morph_max)
            test_captcha.noise = uniform(noise_min, noise_max)
            cci2gf = converters.convert_character_image_to_generic_format
            samples.append(cci2gf(test_captcha))





def train_network(model,
                  params,
                  epochs=32,
                  save_path=None,
                  early_stopper=None,
                  verbose=0):
    """Train given network using samples.

    Inputs
    ------
    model: Network
        Network to train
    params: Dict
        Params of training session.
    epochs: uint, optional
        Epochs to pass before cutoff
    save_path: str or None, optional
        Automatically saves trained network here if not None
    verbose: 0, 1, 2
        Supplied to fit_generator()
    ------
    """



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
             return_value.append('???')
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
