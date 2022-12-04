# -*- coding: UTF-8 -*-
import os
import random

import numpy as np
from PIL import Image
from keras_preprocessing.image import save_img

from GAN.utils.utils import load_model, gen_img, create_sample
from captcha_setting import NUM_CLASSES, LETTER_HEIGHT, LETTER_WIDTH, \
    ALL_CHAR_SET, CGAN_MODEL, MAX_CAPTCHA

path = os.path.join(os.getcwd(), f"../../SavedModels/{CGAN_MODEL}/Generator/weights.h5")
generator = load_model(path)


def save_all(cluster, brightness_factor=1.25):
    # Create some examples
    for char in ALL_CHAR_SET:
        fake = gen_img(char, generator, brightness_factor=brightness_factor, to_rgb=True)
        # Saving synthetic image
        save_img(os.getcwd() + f"/{cluster}_letter_{char}.png", fake)


def create_seq(label=None, brightness_factor=1.0):
    img, label = create_sample(generator, label=label, brightness_factor=brightness_factor)
    img = np.reshape(img, [LETTER_HEIGHT, LETTER_WIDTH * 4])
    img = (img * 255 / np.max(img)).astype('uint8')
    img = Image.fromarray(img, mode='L').resize((100, 40))
    img.save('seq.png')
    print(label)
    with open('seq.txt', 'w') as file:
        file.write(''.join(label))


def create_all_seq(path, label, brightness_factor=1.0):
    img, label = create_sample(generator,
                               label,
                               brightness_factor=brightness_factor,
                               is_res=True)
    img = np.reshape(img, [LETTER_HEIGHT, LETTER_WIDTH * 4])
    img = (img * 255 / np.max(img)).astype('uint8')
    img = Image.fromarray(img, mode='L')  # .resize((100, 40))
    img.save(os.path.join(os.getcwd(), path))


def create_all_rgb_seq(label=None, brightness_factor=1.0):
    img, label = create_sample(generator,
                               label=label,
                               brightness_factor=brightness_factor,
                               to_rgb=True)
    img = np.reshape(img, [LETTER_HEIGHT, LETTER_WIDTH * 4, 3])
    img = (img * 255 / np.max(img)).astype('uint8')
    img = Image.fromarray(img, mode='RGB')  # .resize((100, 40))
    img.save(os.path.join(os.getcwd(), f'../../CNN/data/{"".join(label)}.png'))


def create_random_samples_for_solver(count):
    for i in range(count):
        label = [ALL_CHAR_SET[random.randint(0, NUM_CLASSES - 1)] for _ in range(MAX_CAPTCHA)]
        create_all_seq(path=f'../../CNN/res_data/{"".join(label)}.png',
                       label=label,
                       brightness_factor=1.25)
        if i % 10000 == 0:
            print(i)


if __name__ == '__main__':
    create_random_samples_for_solver(40000)
