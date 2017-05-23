"""
This module implement Introspector class used to obtain intermediate results
for given model.
"""
from collections import deque
from math import ceil

from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

class Introspector:
    def __init__(self, source_model):
        self.parent_model = source_model
        input_layer = source_model.input
        layers = source_model.layers
        self.dummy_models = dict()
        for index, output_layer in enumerate(layers):
            dummy_model = Model(inputs=input_layer,
                                outputs=output_layer.output)
            self.dummy_models[index] = dummy_model
            self.dummy_models[output_layer.name] = dummy_model

    def get_raw_output_for_layer(self,
                                 data_in,
                                 layer_num=0,
                                 layer_name=None):
        """Gets output for respective layer

        Inputs:
        -------
        data_in: 3D array of shape (?, ?, 1)
            Data to be sent
        layer_num: uint
            Index of layer to obtain data from.
                Ignored if layer_name is not None.
        layer_name: str
            Name of layer to obtain data from.
        -------

        Output:
        -------
            Depends on output shape of last layer.
        -------
        """

        chosen_model = self.dummy_models[layer_name or layer_num]
        proper_input = data_in.reshape((1, *chosen_model.input_shape[1:]))
        data_output = chosen_model.predict(proper_input)
        data_shape = data_output.shape[1:]
        if len(data_shape) == 3:
            if data_shape[2] == 1:
                return data_output.reshape(data_shape[1:4])
            return data_output.reshape(data_shape)
        if len(data_shape) == 2:
            if data_shape[1] == 1:
                return data_output.reshape((data_shape[1],))
            return data_output.reshape(data_shape[1:3])
        if len(data_shape) == 1:
            return data_output.reshape(data_shape[0:1])
        print("Unknown data shape")
        return None

    def get_view(self,
                    data_in,
                    fig_size=(10, 10),
                    layer_num=0,
                    layer_name=None,
                    fig_name=None):
        output = self.get_raw_output_for_layer(data_in,
                                               layer_num=layer_num,
                                               layer_name=layer_name)
        if len(output.shape) == 1:
            # Line of outputs
            f, ax = plt.subplots()
            normalized_output = 0.5 + output / np.max(np.abs(output)) / 2
            ax.plot(normalized_output, marker='+')
            ax.set_xlabel("Some ID")
            ax.set_ylabel("Value")
            f.set_figwidth(fig_size[0])
            f.set_figheight(fig_size[1])
            return f
        if len(output.shape) == 2:
            # Simple image
            f, ax = plt.subplots()
            normalized_output = 0.5 + output / np.max(np.abs(output)) / 2
            ax.imshow(normalized_output, cmap='binary')
            f.set_figwidth(fig_size[0])
            f.set_figheight(fig_size[1])
            ax.set_axis_off()
            return f
        if len(output.shape) == 3:
            # Images
            normalized_output = 0.5 + output / np.max(np.abs(output)) / 2
            min_square = ceil(normalized_output.shape[2]**0.5)
            f, ax = plt.subplots(min_square,
                                 min_square)
            fax = ax.flatten()
            # Sigh...
            for index in range(normalized_output.shape[2]):
                image_data = normalized_output[:, :, index]
                plot_data = fax[index].imshow(image_data,
                                              vmin=0.0,
                                              vmax=1.0,
                                              cmap='binary')
                fax[index].set_axis_off()
                # f.colorbar(plot_data, ax=fax[index])
            return f
        print("What")
        return None