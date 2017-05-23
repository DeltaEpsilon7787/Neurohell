# -*- coding: utf-8 -*-
"""
Implements a single class Recognizer that is used externally.
"""

from itertools import chain
from os.path import exists
from collections import deque
from json import load, dump

from CNN import CNN
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np


class Recognizer:
    def __init__(self, classes, config_path, start_anew=False):
        """Init Recognizer class.

        Inputs
        ------
        classes: iterable of characters
            Characters that this network should recognize
        config_path: str
            Path to config file
        start_anew=False: bool, optional
            Create a fresh network or load it from IOPath from config file?
        ------
        """
        with open(config_path, mode='r') as new_params_file:
            new_params = load(new_params_file)
        with open(config_path+'.default', mode='r') as default_params_file:
            default_params = load(default_params_file)

        for param in default_params:
            if param not in new_params:
                new_params[param] = default_params[param]

        self._input_shape = tuple(new_params['input_shape'])
        self._architecture = new_params['architecture_code']

        architecture_data = new_params['architecture']
        optimizer = new_params['optimizer']
        loss = new_params['loss']

        self.IOPath = new_params.get('IOPath')  # Optional
        self.validation_split = new_params['validation_split']
        self.confidence_threshold = new_params['confidence_threshold']

        self.num_classes = len(classes)
        characters = classes
        codes = tuple(range(self.num_classes))
        self._mask = dict(
                chain(
                    zip(characters,
                        codes),
                    zip(codes,
                        characters)
                    )
                )

        stopper_patience = new_params['early_stopper_patience']
        lr_decreaser_patience = new_params['lr_decreaser_patience']
        lr_decreaser_factor = new_params['lr_decreaser_factor']


        self._early_stop = EarlyStopping(monitor='val_loss',
                                         patience=stopper_patience)

        self._lr_decreaser = ReduceLROnPlateau(monitor='val_loss',
                                               factor=lr_decreaser_factor,
                                               patience=lr_decreaser_patience,
                                               verbose=1)

        self._init_network(start_anew, architecture_data, optimizer, loss)

    def _init_network(self,
                      start_anew,
                      architecture_data,
                      optimizer,
                      loss):
        if not start_anew and self.IOPath and exists(self.IOPath):
            self.model = CNN.load_network(self.IOPath)
        else:
            self.model = CNN.generate_network(self._input_shape,
                                              self._architecture,
                                              architecture_data,
                                              optimizer,
                                              loss)

    def train_model(self, data_in, data_out, epochs=32, verbose=0):
        """Train model using given data

        Inputs
        ------
        data_in: 4DArray of uint0 with shape (samples, height, width, 1)
            Samples to be forwarded.
        data_out: 1DArray of char with shape (samples)
            Expected predictions
        epochs: uint, optional
            Train for this amount of epochs
        ------
        """

        # Split to validation and training
        unique = deque()
        for char in data_out:
            if char not in unique:
                unique.append(char)
        samples_per_char = data_out.shape[0] // len(unique)

        va, vb = (0, int(self.validation_split * samples_per_char))

        train_data_in = deque()
        train_data_out = deque()
        val_data_in = deque()
        val_data_out = deque()

        for indx, _ in enumerate(data_in):
            real_indx = indx % samples_per_char
            if va <= real_indx < vb:
                val_data_in.append(data_in[indx])
                val_data_out.append(data_out[indx])
            else:
                train_data_in.append(data_in[indx])
                train_data_out.append(data_out[indx])

        train_data_in = np.array(train_data_in)
        train_data_out = CNN.chars_to_classes(train_data_out,
                                              self._mask,
                                              self.num_classes)
        val_data_in = np.array(val_data_in)
        val_data_out = CNN.chars_to_classes(val_data_out,
                                            self._mask,
                                            self.num_classes)

        CNN.train_network(self.model,
                          train_data_in,
                          train_data_out,
                          epochs=epochs,
                          val_data_in=val_data_in,
                          val_data_out=val_data_out,
                          save_path=self.IOPath,
                          early_stopper=self._early_stop,
                          verbose=verbose)

    def recognize(self, data_in):
        """Recognize given image.

        Inputs
        ------
        data_in: 4DArray of uint0 with shape(?, height, width, 1)
            Batch to recognize
        ------

        Returns
        -------
            List of characters
        -------
        """

        return CNN.pass_data(self.model,
                             data_in,
                             self._mask,
                             self.num_classes,
                             self.confidence_threshold)
