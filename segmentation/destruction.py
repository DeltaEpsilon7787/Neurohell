from collections import deque
from operator import mul

import numpy as np
import scipy.ndimage as nd
from skimage.morphology import remove_small_objects

import segmentation.utility as utils


def destroy_small_bridges(image, width=1):
    data = utils.image2data(image)
    fullness_data = np.sum(data, axis=0)
    mask = fullness_data >= width
    data[:, np.invert(mask)] = 0
    bridgeless_image = utils.data2image(data)
    return bridgeless_image


def destroy_small_objects(image, min_size):
    data = utils.image2data(image)
    eroded = remove_small_objects(data, min_size=mul(*min_size))
    eroded_image = utils.data2image(eroded)
    return eroded_image


def destroy_small_images(image_set, min_area):
    filtered_set = (image
                    for image in image_set
                    if mul(*image.size) >= mul(*min_area))
    return deque(filtered_set)


def destroy_holes(image, coeff=2):
    data = utils.image2data(image)
    dilated = nd.binary_dilation(data, iterations=coeff)
    eroded = nd.binary_erosion(dilated, iterations=coeff)
    filled_image = utils.data2image(eroded)
    return filled_image
