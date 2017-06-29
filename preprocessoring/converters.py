from collections import deque

import numpy as np

import segmentation.destruction as destruction
import segmentation.splitting as splitting
import segmentation.utility as utils


def process_image(image, params):
    noise_filter_size = params.get('noise_filter_size', (30, 30))
    holes_removal_coeff = params.get('holes_removal_coeff', 2)

    image = image.quantize(2)  # Binarization
    image = image.crop(image.getbbox())
    # image = destruction.destroy_small_objects(image, noise_filter_size)  # Noise removal

    advanced_segmentation = params.get('advanced_segmentation_mode', 0)
    expected_characters_amount = params.get('expected_character_amount', 5)
    if advanced_segmentation:
        raise NotImplementedError()

    image_set = splitting.split_evenly(image, expected_characters_amount)
    image_set = deque((
                    destruction.destroy_holes(image, holes_removal_coeff)
                    for image in image_set))
    image_set = destruction.destroy_small_images(image_set, noise_filter_size)
    image_set = utils.unrotate_image_set(image_set)
    return image_set


def image2CNNdata(image):
    data = np.array(image).astype(np.uint8)
    mask = data > 127
    data[mask] = 0
    data[np.invert(mask)] = 1
    return data
