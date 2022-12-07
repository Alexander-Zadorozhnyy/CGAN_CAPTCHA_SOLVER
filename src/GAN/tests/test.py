# -*- coding: UTF-8 -*-
import os

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from keras_preprocessing.image import save_img

from src.GAN.utils.utils import gen_img, create_sample
from captcha_setting import LETTER_HEIGHT, LETTER_WIDTH, ALL_CHAR_SET, CGAN_MODEL


path = os.path.join(os.getcwd(), f"trained_models/{CGAN_MODEL}/Generator/generator.h5")
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


if __name__ == '__main__':
    pass
