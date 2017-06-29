from collections import deque

import numpy as np
import scipy.ndimage as nd
from PIL import Image

import segmentation.utility as utils


def split_by_connectivity(image):
    data = utils.image2data(image)
    objects, characters_amt = nd.label(data)

    index_sequence = deque()  # Order must be preserved
    for index in range(1, characters_amt+1):
        mask = objects == index
        indices = np.argwhere(mask)[:, 1]
        left_minimum = np.min(indices)
        index_sequence.append((index, left_minimum))
    index_sequence = sorted(index_sequence, key=lambda m: m[1])
    index_sequence = deque((m[0] for m in index_sequence))

    characters = deque()
    for index in index_sequence:
        mask = objects == index
        characters.append((255*(objects * mask)).astype(np.uint8))
    character_images = deque()
    for character in characters:
        image = Image.fromarray(character, mode='L')
        character_images.append(image.crop(image.getbbox()).quantize(2))
    return character_images


def split_evenly(image, n):
    proper_image = image.crop(image.getbbox())
    data = utils.image2data(proper_image)
    split_point = data.shape[1] // n
    for i in range(1, n):
        data[:, split_point*i] = 0
    split_image = utils.data2image(data)
    return split_by_connectivity(split_image)
