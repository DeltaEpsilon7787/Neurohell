import numpy as np

def image2CNNdata(image):
    data = np.expand_dims(np.array(image), axis=2)
    data = 1 - data / 255
    return data
