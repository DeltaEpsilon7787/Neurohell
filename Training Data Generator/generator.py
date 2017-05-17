# -*- coding: utf-8 -*-

import argparse as ap
import os
from string import ascii_letters, digits
from random import choice
from shutil import rmtree

import claptcha
import numpy as np
from scipy.ndimage.filters import sobel
from PIL import Image

parser = ap.ArgumentParser()
parser.add_argument('samples', type=int)
parser.add_argument('-r', '--restart', action='store_true')
parser.add_argument('--characters', type=str, action='store', default=ascii_letters+digits)
parser.add_argument('--width', type=int, action='store', default=40)
parser.add_argument('--height', type=int, action='store', default=60)
parser.add_argument('--morphmin', type=float, action='store', default=0.2)
parser.add_argument('--morphmax', type=float, action='store', default=0.7)

args = parser.parse_args()

save_directory = "../Samples"
font_directory = "./Fonts"
size = (args.width, args.height)
fonts = os.listdir(font_directory)

if args.restart:
    while os.path.exists(save_directory):
        rmtree(save_directory, ignore_errors=True)

for char in args.characters:
    char_code = str(ord(char))
    new_dir = os.path.join(save_directory, char_code)

    while not os.path.exists(new_dir):
        os.makedirs(new_dir)

    files_already_present = os.listdir(new_dir)

    for i in range(args.samples):
        new_name = str(i)+'.bmp'
        if new_name in files_already_present:
            continue
        random_font = os.path.join(font_directory, choice(fonts))
        char_image = claptcha.Claptcha(char,
                                       random_font,
                                       margin=(0, 0),
                                       size=(200,250),
                                       injectLine=False,
                                       morph_min_coeff=args.morphmin,
                                       morph_max_coeff=args.morphmax)

        char_data = np.array(char_image.image[1].convert('L'))

        sobel_data = sobel(char_data)

        # threshold = 100
        # threshold_mask = sobel_data < threshold
        # sobel_data[threshold_mask] = 0
        # sobel_data[np.invert(threshold_mask)] = 255

        sobel_image = Image.fromarray(sobel_data, mode='L').resize(size)
        new_file = os.path.join(new_dir, new_name)
        sobel_image.save(new_file)
        while not os.path.exists(new_file):
            pass
