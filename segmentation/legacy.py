import numpy as np
import scipy.ndimage as nd
from PIL import Image


def split_by_character_width(image, width=10):
    data = np.array(image)

    mask = np.ones_like(data)
    horizontal_mask = (mask.cumsum(axis=1)-1) % width
    vertical_mask = (mask.cumsum(axis=0)-1) % width

    h_coords = np.argwhere(horizontal_mask == 0)
    v_coords = np.argwhere(vertical_mask == 0)

    h_table = np.zeros(h_coords.shape[0]).astype(np.uint8)
    v_table = np.zeros(v_coords.shape[0]).astype(np.uint8)

    horizontal_width_image = data.copy()
    vertical_width_image = data.copy()

    for i, p in enumerate(h_coords):
        x, y = p
        hits = 0
        for delta in range(width):
            if data[x+delta
                    if x+delta < data.shape[0]
                    else data.shape[0]-1, y]:
                hits += 1
                if hits >= width:
                    h_coords[i] = 1

    for i, p in enumerate(v_coords):
        x, y = p
        hits = 0
        for delta in range(width):
            if data[x, y+delta
                    if y+delta < data.shape[1]
                    else data.shape[1]-1]:
                hits += 1
                if hits >= width:
                    v_coords[i] = 1

    for v, p in zip(h_table, h_coords):
        x, y = p
        horizontal_width_image[x:x+width, y] *= v

    for v, p in zip(v_table, v_coords):
        x, y = p
        vertical_width_image[x, y:y+width] *= v

    return (horizontal_width_image, vertical_width_image)


def destroy_small_bridges(image, char_width=5):
    data = np.array(image).astype(np.bool)
    new_data = nd.minimum_filter(data, footprint=np.ones((char_width, char_width)))
    new_image = Image.fromarray((255*new_data).astype(np.uint8))
    return new_image


def estimate_character_width(image, thickness=2):
    data = np.array(image)
    data = np.invert(data)
    features, num_features = nd.label(data)
    def get_dimensions_and_area_of_feature(feature_index):
        feature = features == feature_index
        area = np.count_nonzero(feature) # This is easy enough
        feature_regions = np.argwhere(feature)
        left_top_x = np.min(feature_regions[:, 0])
        left_top_y = np.min(feature_regions[:, 1])
        right_bottom_x = np.max(feature_regions[:, 0])
        right_bottom_y = np.max(feature_regions[:, 1])

        width, height = right_bottom_x - left_top_x + 1, right_bottom_y - left_top_y + 1
        return (area, width, height)

    def is_triangle(feature):
        pass
