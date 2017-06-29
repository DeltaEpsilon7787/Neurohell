from collections import deque
from operator import mul

import numpy as np
from PIL import Image


def image2data(image):
    # if image.mode != 'P' or len(image.getcolors()) != 2:
    #    raise TypeError("Wrong image mode or not binary image")
    if image.mode == 'L':
        return (np.array(image) // 255).astype(np.uint8)
    if image.mode == 'P':
        return np.array(image).astype(np.uint8)


def data2image(data):
    if np.max(data) == 1 or np.min(data) == 0:
        return Image.fromarray((255*data).astype(np.uint8))
    else:
        return image.fromarray(data.astype(np.uint8), mode='L')


def insert_boundaries(image, boundary_set):
    data = np.array(image)
    data[data > 0] = 255
    coordinate_stack = np.stack(boundary_set)
    for i in range(coordinate_stack.shape[0]):
        data[coordinate_stack[i, :, 0], coordinate_stack[i, :, 1]] = 0
    boundary_image = Image.fromarray(data)
    return boundary_image


def unrotate_image_set(image_set):
    angles = tuple(range(-20, 20, 1))
    unrotated_image_set = deque()
    for image in image_set:
        crop_to_char = lambda image: image.crop(image.getbbox())
        rotated_images = ((crop_to_char(image.rotate(angle, expand=True)),
                           angle) for angle in angles)
        rotated_images = [
                (mul(*image.size), image, angle)
                for image, angle in rotated_images]

        min_area = min(rotated_images, key=lambda t: t[0])[0]
        candidates = [
                t
                for t in rotated_images
                if t[0] <= min_area]
        min_angle = min(candidates, key=lambda t: t[2])[2]
        unrotated = [
                t[1]
                for t in candidates
                if t[2] == min_angle][0]  # Guaranteed to be unique
        unrotated_image_set.append(unrotated)
    return unrotated_image_set
