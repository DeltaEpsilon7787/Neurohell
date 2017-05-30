# -*- coding: utf-8 -*-
"""
Implements a single class Recognizer that is used externally.
"""

from collections import deque
from itertools import chain
from json import load, dump
from os.path import exists
from string import ascii_uppercase, digits

import numpy as np
from CNN import CNN
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class Recognizer:
    def __init__(self, config_path, start_anew=False, verbosity=0):
        """Init Recognizer class.

        Inputs
        ------
        config_path: str
            Path to config file
        start_anew: bool, default False
            Create a fresh network or load it from IOPath from config file?
        ------
        """
        self.verbosity = verbosity
        with open(config_path, mode='r') as new_params_file:
            new_params = load(new_params_file)
        self._IOPath = new_params.get('IOPath', './CNN.neuro')

        input_shape = tuple(new_params['input_shape'])
        architecture = new_params['architecture_code']
        architecture_data = new_params['architecture']
        self._optimizer = new_params.get('optimizer', 'adam')
        self._loss = new_params.get('loss', 'categorical_crossentropy')
        model_params = {'input_shape': input_shape,
                        'architecture': architecture,
                        'architecture_data': architecture_data,
                        'optimizer': self._optimizer,
                        'loss': self._loss}
        self._init_network(model_params,
                           start_anew=start_anew)

        self.confidence_threshold = new_params.get('confidence_threshold', 0.9)
        self._characters = new_params.get('characters', ascii_uppercase+digits)
        self._create_mask()

        morph_min = new_params.get('morphological_coefficient_min', 0.2)
        morph_max = new_params.get('morphological_coefficient_max', 0.7)
        noise_min = new_params.get('noise_coefficient_min', 0.0)
        noise_max = new_params.get('noise_coefficient_max', 1.0)
        line_prob = new_params.get('line_prob', 0.5)
        font_directory = new_params.get('font_directory', r'./Fonts')
        self._samples_per_batch = new_params.get('samples_per_batch', 50)
        self._test_samples_per_batch = new_params.get('test_samples_per_batch', 10)
        training_params = {'characters': self._characters,
                           'morphological_coefficient_min': morph_min,
                           'morphological_coefficient_max': morph_max,
                           'noise_coefficient_min': noise_min,
                           'noise_coefficient_max': noise_max,
                           'line_probability': line_prob,
                           'font_directory': font_directory,
                           'samples_per_batch': self._samples_per_batch,
                           'mask': self._mask,
                           'num_classes': self._num_classes,
                           'expected_shape': input_shape,
                           'finite': False}
        self._assign_training_data_generator(training_params)
        training_params['samples_per_batch'] = self._test_samples_per_batch
        self._assign_test_data_generator(training_params)

        stopper_patience = new_params.get('early_stopper_patience', 8)
        lr_decreaser_patience = new_params.get('lr_decreaser_patience', 2)
        lr_decreaser_factor = new_params.get('lr_decreaser_factor', 0.5)
        lr_cooldown = new_params.get('lr_cooldown', 5)
        early_stopper = EarlyStopping(monitor='val_loss',
                                      patience=stopper_patience)
        lr_decreaser = ReduceLROnPlateau(monitor='val_loss',
                                         factor=lr_decreaser_factor,
                                         patience=lr_decreaser_patience,
                                         verbose=self.verbosity,
                                         cooldown=lr_cooldown)
        model_saver = ModelCheckpoint(self._IOPath,
                                      monitor='val_loss',
                                      verbose=self.verbosity,
                                      save_best_only=True)
        self._callbacks = [early_stopper, lr_decreaser, model_saver]

    def _init_network(self,
                      model_params,
                      start_anew=False):
        if not start_anew and self._IOPath and exists(self._IOPath):
            self.model = CNN.load_network(self._IOPath)
        else:
            self.model = CNN.generate_network(**model_params)

    def _create_mask(self):
        self._num_classes = len(self._characters)
        codes = tuple(range(len(self._characters)))
        self._mask = dict(
                chain(
                    zip(self._characters,
                        codes),
                    zip(codes,
                        self._characters)
                    )
                )

    def _assign_training_data_generator(self,
                                        params):
        self._training_data_generator = CNN.create_data_generator(**params)

    def _assign_test_data_generator(self,
                                    params):
        self._test_data_generator = CNN.create_data_generator(**params)

    def train_model(self,
                    epochs=32,
                    verbose=0,
                    sample_source=None,
                    validation_source=None):
        """Trains this model for given amount of epochs

        Inputs
        ------
        epochs: uint, default 32
            Amount of epochs to train for at most
        verbose: 0 or 1 or 2, default 0
            Verbosity level
        sample_source: str, default None
            If provided, this will be the source of training data
                otherwise data will be generated
        validation_source: str, default None
            Same as sample_source,.but for testing
        ------

        Returns
        -------
        History of training
        -------
        """
        training_generator = (
                CNN.create_stored_data_generator(sample_source,
                                                 self._mask,
                                                 self._num_classes)
                if sample_source
                else self._training_data_generator)

        validation_generator = (
                CNN.create_stored_data_generator(validation_source,
                                                 self._mask,
                                                 self._num_classes)
                if validation_source
                else self._test_data_generator)

        training_params = {'generator': training_generator,
                           'steps_per_epoch': self._samples_per_batch,
                           'callbacks': self._callbacks,
                           'validation_data': validation_generator,
                           'validation_steps': self._test_samples_per_batch}
        return CNN.train_network(self.model,
                                 training_params,
                              epochs=epochs,
                              verbose=verbose,
                              save_path=self._IOPath)

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
                             self._num_classes,
                             self.confidence_threshold)
