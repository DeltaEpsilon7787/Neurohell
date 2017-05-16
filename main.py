# -*- coding: utf-8 -*-

import CNN

from functools import reduce
from operator import add
from os import listdir, path

import numpy as np
from PIL import Image
from string import ascii_letters, digits

try:
    len(samples)
except:
    all_chars = ""
    samples = []
    answers = []
    path1 = r".\Samples"

    for char_code in listdir(path1):
        char = chr(int(char_code))
        path2 = path.join(path1, char_code)
        for file in listdir(path2):
            new_data = np.array(Image.open(path.join(path2, file)).convert('L'))
            new_data = 1 - new_data / 255  # Grayscale
            # threshold = 0
            # mask = new_data < threshold
            # new_data[mask] = 0
            # new_data[np.invert(mask)] = 1
            samples.append(new_data)
            answers.append(char)

    samples = np.array(samples)
    samples = samples.reshape((*samples.shape[:3], 1))
    answers = np.array(answers)
    all_chars = reduce(add, np.unique(answers), "")

alpha = CNN.Recognizer.Recognizer(all_chars, 'CNN.params', True)
beta = CNN.Introspection.Introspector(alpha.model)

alpha.train_model(samples, answers, epochs=128, verbose=1)


#==============================================================================
# dist = samples.shape[0] // len(all_chars)
#
# beta.get_view(samples[0], layer_num=0)
# beta.get_view(samples[0], layer_num=2)
# beta.get_view(samples[0], layer_num=4)
# beta.get_view(samples[0], layer_num=7)
#
# beta.get_view(samples[dist], layer_num=0)
# beta.get_view(samples[dist], layer_num=2)
# beta.get_view(samples[dist], layer_num=4)
# beta.get_view(samples[dist], layer_num=7)
#
# beta.get_view(samples[2*dist], layer_num=0)
# beta.get_view(samples[2*dist], layer_num=2)
# beta.get_view(samples[2*dist], layer_num=4)
# beta.get_view(samples[2*dist], layer_num=7)
#==============================================================================

# gamma = alpha.recognize(samples[:2000])
