# -*- coding: UTF-8 -*-
import os
import random

import numpy as np
from numpy.random import multinomial
import tensorflow as tf
from PIL import Image, ImageEnhance

from src.GAN.models.generator import Generator
from src.GAN.utils.one_hot_encoding import encode
from captcha_setting import CGAN_LATENT_DIM, NUM_CLASSES, LETTER_HEIGHT, LETTER_WIDTH, \
    ALL_CHAR_SET, MAX_CAPTCHA, CGAN_MODEL, IMAGE_WIDTH, IMAGE_HEIGHT


def load_model(path):
    # Params advertisement
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    generator_in_channels = CGAN_LATENT_DIM + NUM_CLASSES
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Create generator
    generator = Generator(in_channels=generator_in_channels, optimizer=g_optimizer, loss_fn=loss_fn,
                          height=LETTER_HEIGHT,
                          width=LETTER_WIDTH)

    # Load pretrained weights
    generator.model.load_weights(path)

    return generator


def evaluate_sizes(summa, count: int):
    return multinomial(summa, [1 / np.float32(count)] * count)


def customize_image(img, brightness_factor, to_rgb=False, size: tuple = None):
    img = np.reshape(img, [LETTER_HEIGHT, LETTER_WIDTH])
    img = (img * 255 / np.max(img)).astype('uint8')
    img = Image.fromarray(img, mode='L')

    if size is not None:
        img = img.resize(size)

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    if to_rgb:
        img = img.convert("RGB")

    # img.save('letter.png')
    # img = np.asarray(img).astype('float32') / 255
    return img


def gen_img(name, generator, brightness_factor=1.0, to_rgb=False, size=None):
    fake = None
    # Sample noise for testing.
    noise = tf.random.normal(shape=(1, CGAN_LATENT_DIM))
    # label = encode("H")
    label = encode(str(name))
    # Prepare label for generator
    label = np.reshape(label, (1, NUM_CLASSES))
    label = tf.convert_to_tensor(label, np.float32)

    # Combine the noise and the labels and run inference with the generator.
    noise_and_labels = tf.concat([noise, label], 1)

    if isinstance(generator, type(tf.keras.Sequential())):
        fake = generator(noise_and_labels)
    else:
        fake = generator.generate(noise_and_labels)

    if brightness_factor != 1 or to_rgb or size is not None:
        fake = customize_image(fake, brightness_factor=brightness_factor, to_rgb=to_rgb, size=size)

    return fake
    # if to_rgb:
    #     return np.reshape(fake, [LETTER_HEIGHT, LETTER_WIDTH, 3])
    # return np.reshape(fake, [LETTER_HEIGHT, LETTER_WIDTH, 1])


def create_sample(generator, label=None, brightness_factor=1.0, to_rgb=False, is_res=False):
    images = None
    if label is None:
        label = [ALL_CHAR_SET[random.randint(0, NUM_CLASSES - 1)] for _ in range(MAX_CAPTCHA)]

    if is_res:
        sizes = evaluate_sizes(IMAGE_WIDTH, MAX_CAPTCHA)
        images = [gen_img(label[x],
                          generator,
                          brightness_factor,
                          to_rgb,
                          (sizes[x], IMAGE_HEIGHT)
                          ) for x in range(len(label))]
    else:
        images = [gen_img(x,
                          generator,
                          brightness_factor,
                          to_rgb
                          ) for x in label]

    seq = Image.new('RGB' if to_rgb else 'L', (IMAGE_WIDTH, IMAGE_HEIGHT))

    x_offset = 0
    for img in images:
        seq.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    seq = np.asarray(seq).astype('float32') / 255
    if to_rgb:
        return np.reshape(seq, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    return np.reshape(seq, [IMAGE_HEIGHT, IMAGE_WIDTH, 1]), label
    # return np.concatenate(images, axis=1), label


if __name__ == '__main__':
    # print(evaluate_sizes(96, 4))
    path = os.path.join(os.getcwd(), f"../../SavedModels/{CGAN_MODEL}/Generator/weights.h5")
    generator = load_model(path)
    img, label = create_sample(generator, label='2LUG', brightness_factor=1.25, is_res=True)
    img = np.reshape(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img = (img * 255 / np.max(img)).astype('uint8')
    img = Image.fromarray(img, mode='L').resize((100, 40))
    img.save('seq.png')
