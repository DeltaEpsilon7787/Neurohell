# -*- coding: utf-8 -*-

import CNN

alpha = CNN.Recognizer.Recognizer('CNN.json', start_anew=True, verbosity=1)
beta = CNN.Introspection.Introspector(alpha.model)

gamma = alpha.train_model(epochs=100, verbose=1)


def handwritten_recognition(filename):
    from preprocessor.converters import image2CNNdata
    from PIL import Image
    return alpha.recognize(image2CNNdata(Image.open(filename).convert('L')).reshape((1, 60, 40, 1)))


def gen_sample_set():
    gen = alpha._training_data_generator
    s, a = gen.__next__()
    return s


def transform_and_output(sample):
    from PIL import Image
    from numpy import uint8
    return Image.fromarray((255 - 255*sample.reshape((60,40))).astype(uint8))
