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
            with Image.open(path.join(path2, file)) as img:
                new_data = np.array(img.convert('L'))
                new_data = 1 - new_data / 255
                threshold = 0.5
                mask = new_data < threshold
                new_data[mask] = -1
                new_data[np.invert(mask)] = 1
                samples.append(new_data)
                answers.append(char)

    samples = np.array(samples)
    samples = samples.reshape((*samples.shape[:3], 1))
    answers = np.array(answers)
    all_chars = reduce(add, np.unique(answers), "")

alpha = CNN.Recognizer.Recognizer(all_chars, 'CNN.params', True)
beta = CNN.Introspection.Introspector(alpha.model)

gamma = alpha.train_model(samples, answers, epochs=1024, verbose=1)


def test(char, attempts):
    from claptcha import Claptcha
    from scipy.ndimage.filters import sobel
    test_samples = []
    for i in range(attempts):
        test_image = Claptcha(char, r"C:/Windows/Fonts/arial.ttf",size=(200,250)).image[1]
        im_data = np.array(test_image.convert('L'))
        im_data = sobel(im_data)
        img2 = Image.fromarray(im_data, mode='L').resize((40,60))
        im_data = np.array(img2)
        test_samples.append(im_data)
    test_samples = np.array(test_samples)
    test_samples = test_samples.reshape((*test_samples.shape[:3], 1))
    test_answers = alpha.recognize(test_samples[:])
    correct = np.nonzero(test_answers == char)[0].shape[0] / attempts
    return test_samples

# dist = samples.shape[0] // len(all_chars)
#
# beta.get_view(samples[0], layer_num=0)
# beta.get_view(samples[0], layer_num=2)
# beta.get_view(samples[0], layer_num=4)
# beta.get_view(samples[0], layer_num=7)
#`
# beta.get_view(samples[dist], layer_num=0)
# beta.get_view(samples[dist], layer_num=2)
# beta.get_view(samples[dist], layer_num=4)
# beta.get_view(samples[dist], layer_num=7)
#
# beta.get_view(samples[2*dist], layer_num=0)
# beta.get_view(samples[2*dist], layer_num=2)
# beta.get_view(samples[2*dist], layer_num=4)
# beta.get_view(samples[2*dist], layer_num=7)


# gamma = alpha.recognize(samples[:2000])
