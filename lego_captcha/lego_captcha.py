# -*- coding: utf-8 -*-
"""
Created on Sun May 28 02:08:47 2017

@author: Editor
"""

from collections import deque
from itertools import accumulate, chain
from functools import reduce
from operator import add
from random import randint, uniform

from PIL import Image, ImageChops, ImageDraw, ImageFont


def generate_captcha(text,
                     size,
                     font,
                     font_size=200,
                     max_angle=10,
                     max_horizontal_deviation=0,
                     max_vertical_deviation=0,
                     foreground_iterations=1000,
                     background_iterations=10,
                     foreground_bounds=(0, 200),
                     background_bounds=(180, 255),
                     foreground_size_limit=10,
                     background_size_limit=5):

    foreground_limit = (-foreground_size_limit, foreground_size_limit)
    background_limit = (-background_size_limit, background_size_limit)
    repeat_func_n_times = lambda func, args, n: tuple(func(*args) for _ in range(n))
    background = Image.new(mode='RGB', size=size, color=(255, 255, 255))
    foreground = Image.new(mode='RGB', size=size, color=(255, 255, 255))
    font_object = ImageFont.truetype(font, size=font_size)

    foreground_draw = ImageDraw.Draw(foreground)
    background_draw = ImageDraw.Draw(background)

    """Create random background noise and prepare random gap noise"""
    for i in range(foreground_iterations):
        r1x = randint(0, size[0])
        r1y = randint(0, size[1])
        r2x = r1x + randint(*foreground_limit)
        r2y = r1y + randint(*foreground_limit)
        random_rectangle = (r1x, r1y, r2x, r2y)
        random_color = repeat_func_n_times(randint, foreground_bounds, 3)
        foreground_draw.rectangle(xy=random_rectangle,
                                  outline=random_color,
                                  fill=random_color)

    for i in range(background_iterations):
        r1x = randint(0, size[0])
        r1y = randint(0, size[1])
        r2x = r1x + randint(*background_limit)
        r2y = r1y + randint(*background_limit)
        random_rectangle = (r1x, r1y, r2x, r2y)
        random_color = repeat_func_n_times(randint, background_bounds, 3)
        background_draw.rectangle(xy=random_rectangle,
                                  outline=random_color,
                                  fill=random_color)

    text_size = foreground_draw.textsize(text, font_object)
    text_images = deque()
    for character in text:
        character_image = Image.new(mode='RGB', size=text_size, color=(0, 0, 0))
        character_draw = ImageDraw.Draw(character_image)
        character_draw.text(xy=(0, 0),
                            text=character,
                            fill=(255, 255, 255),
                            font=font_object)
        character_center = character_draw.textsize(character, font_object)
        character_center = (character_center[0] // 2, character_center[1] // 2)
        character_image = character_image.rotate(randint(-max_angle, max_angle),
                                                 expand=True,
                                                 center=character_center)
        character_image = character_image.crop(character_image.getbbox())
        text_images.append(character_image)

    paste_boxes = deque()
    cur_x = 0
    for character_image in text_images:
        paste_boxes.append((cur_x, 0,
                            cur_x+character_image.size[0], character_image.size[1]))
        cur_x += character_image.size[0]

    translated_boxes = deque()
    for box in paste_boxes:
        horizontal_deviation = randint(-max_horizontal_deviation, max_horizontal_deviation)
        vertical_deviation = randint(-max_vertical_deviation, max_vertical_deviation)

        new_box = (box[0]+horizontal_deviation,
                   box[1]+vertical_deviation,
                   box[2]+horizontal_deviation,
                   box[3]+vertical_deviation)

        translated_boxes.append(new_box)

    min_x = reduce(lambda r, g: min(r, g[0], g[2]), translated_boxes, 0)
    min_y = reduce(lambda r, g: min(r, g[1], g[3]), translated_boxes, 0)

    normalized_boxes = deque()
    for box in translated_boxes:
        new_box = (box[0]-min_x,
                   box[1]-min_y,
                   box[2]-min_x,
                   box[3]-min_y)
        normalized_boxes.append(new_box)

    max_x = reduce(lambda r, g: max(r, g[0], g[2]), normalized_boxes, 0)
    max_y = reduce(lambda r, g: max(r, g[1], g[3]), normalized_boxes, 0)

    text_image = Image.new(mode='RGB', size=(max_x, max_y), color=(0, 0, 0))
    for box, image in zip(normalized_boxes, text_images):
        text_image.paste(im=image, box=box, mask=image.convert('1'))

    text_image = text_image.resize(size)
    colorful_text = ImageChops.multiply(foreground, text_image)
    final_image = ImageChops.composite(colorful_text, background, mask=text_image.convert('1'))
    return final_image