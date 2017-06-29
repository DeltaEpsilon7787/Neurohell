# -*- coding: utf-8 -*-

# import CNN

# alpha = CNN.Recognizer.Recognizer('CNN.json', start_anew=False, verbosity=1)
# beta = CNN.Introspection.Introspector(alpha.model)

# gamma = alpha.train_model(epochs=100, verbose=1)


from random import sample
from string import ascii_uppercase, digits
from lego_captcha.lego_captcha import generate_captcha
from preprocessoring.converters import process_image

import os
import CNN
alpha = CNN.Recognizer.Recognizer('CNN.json', start_anew=False, verbosity=1)

"""
full = ascii_uppercase+digits
count = 0
id_counter = 0
while count < 50:
    random_string = "".join(sample(full, 5))
    captcha_image = generate_captcha(random_string, (250,50), "C:/Windows/Fonts/arial.ttf")
    images = process_image(captcha_image, {})
    if len(images) == 5: count += 1
    else: continue
    for i, image in enumerate(images):
        fname = "./Test/"+random_string[i]+str(id_counter)+'.jpg'
        image.resize((20,30)).save(fname)
        while not os.path.exists(fname):
            pass
        id_counter += 1
"""

alpha.train_model(sample_source="./Samples", validation_source="./Test", verbose=1)