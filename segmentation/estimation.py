from collections import deque, Counter

import numpy as np


def estimate_character_thickness(image, inclusion_limit=2):
    data = np.array(image)
    runs = deque()
    for row in data:
        counter = 0
        isSequence = False
        for value in row:
            if not isSequence and value:
                isSequence = True
            if isSequence and not value:
                isSequence = False
                counter += 1
        runs.append(counter)

    runs = list(enumerate(runs))
    max_run = max(runs, key=lambda v: v[1])[1]
    best_runs = [run for run in runs if max_run - run[1] <= inclusion_limit]

    run_data = deque()
    for run in best_runs:
        counter = 0
        isSequence = False
        for value in data[run[0]]:
            if not isSequence and value:
                isSequence = True
            if isSequence and value:
                counter += 1
            if isSequence and not value:
                isSequence = False
                run_data.append(counter)
                counter = 0

    run_data = np.array(run_data)
    anomalies_smoothed = run_data ** 0.5
    estimated_thickness = (anomalies_smoothed.sum() / anomalies_smoothed.shape[0])**2

    return round(estimated_thickness)


def estimate_captcha_length(image, scanning_line_y, thickness):
    data = np.array(image).astype(np.bool).astype(np.uint8)
    length = np.sum(data[scanning_line_y, :]) / thickness / 2
    return round(length)


def estimate_linear_boundaries(image, maximum_connection=0.5, angle_limit=25):
    # Boundary is a line that separates one part from another
    data = np.array(image).astype(np.bool)
    image_height = data.shape[0]

    sin_values = np.sin(np.deg2rad(np.arange(-angle_limit, angle_limit, 0.1)))
    delta_x = np.floor(image_height*sin_values).astype(np.int)
    delta_x = np.array(tuple(Counter(delta_x).keys()))

    finds = deque()
    for x1 in range(data.shape[1]):
        best_boundary = None
        best_hit_value = 0
        for dx in delta_x:
            x2 = x1 + dx
            if x2 < 0 or x2 >= data.shape[1]:
                continue
            line_ys = np.arange(0, image_height)
            line_xs = np.linspace(x1, x2, image_height).astype(np.int)
            boundary_line_coords = np.vstack((line_ys, line_xs)).T
            boundary_line_coords[:, 1] -= 1
            boundary_1 = boundary_line_coords.copy()
            boundary_line_coords[:, 1] += 2
            boundary_2 = boundary_line_coords.copy()
            boundary_line_coords[:, 1] -= 1
            illegal_coords_1 = np.logical_or(boundary_1[:, 1] < 0,
                                             boundary_1[:, 1] >= data.shape[1])
            illegal_coords_2 = np.logical_or(boundary_2[:, 1] < 0,
                                             boundary_2[:, 1] >= data.shape[1])
            boundary_1 = boundary_1[np.invert(illegal_coords_1), :]
            boundary_2 = boundary_2[np.invert(illegal_coords_2), :]

            # Boundaries were defined, now evaluate them.
            boundary_1_values = data[boundary_1[:, 0], boundary_1[:, 1]]
            boundary_2_values = data[boundary_2[:, 0], boundary_2[:, 1]]
            if boundary_1_values.shape[0] != boundary_2_values.shape[0]:
                continue
            interconnection_hits = np.logical_and(boundary_1_values,
                                                  boundary_2_values)
            boundary_hits = np.logical_xor(boundary_1_values,
                                           boundary_2_values)

            improper_hits = np.count_nonzero(interconnection_hits)
            proper_hits = np.count_nonzero(boundary_hits)

            """
            if improper_hits < image_height*maximum_connection:
                continue
            """

            proper_hits /= max(improper_hits, 1)
            if proper_hits > best_hit_value:
                best_boundary = boundary_line_coords
                best_hit_value = proper_hits

        if best_hit_value < image_height*maximum_connection:
            continue
        finds.append(best_boundary)
    finds = deque(f for f in finds if f is not None)
    return finds
