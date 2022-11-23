import os
import random
from math import floor

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance

from GAN.models.Generator import Generator
from GAN.utils.one_hot_encoding import encode
from captcha_setting import LATENT_DIM, NUM_CLASSES, LETTER_HEIGHT, LETTER_WIDTH, ALL_CHAR_SET, MAX_CAPTCHA, CGAN_MODEL, \
    IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS


def load_model(path):
    # Params advertisement
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    generator_in_channels = LATENT_DIM + NUM_CLASSES
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Create generator
    generator = Generator(in_channels=generator_in_channels, optimizer=g_optimizer, loss_fn=loss_fn,
                          height=LETTER_HEIGHT,
                          width=LETTER_WIDTH)

    # Load pretrained weights
    generator.model.load_weights(path)

    return generator


def customize_image(img, brightness_factor):
    img = np.reshape(img, [LETTER_HEIGHT, LETTER_WIDTH])
    img = (img * 255 / np.max(img)).astype('uint8')
    img = Image.fromarray(img, mode='L')

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    img = np.asarray(img).astype('float32')
    return img


def gen_img(name, generator, brightness_factor=1.0):
    # Sample noise for testing.
    noise = tf.random.normal(shape=(1, LATENT_DIM))
    # label = encode("H")
    label = encode(str(name))
    # Prepare label for generator
    label = np.reshape(label, (1, NUM_CLASSES))
    label = tf.convert_to_tensor(label, np.float32)

    # Combine the noise and the labels and run inference with the generator.
    noise_and_labels = tf.concat([noise, label], 1)
    fake = generator.generate(noise_and_labels)

    if brightness_factor != 1:
        fake = customize_image(fake, brightness_factor=brightness_factor)

    fake = np.reshape(fake, [LETTER_HEIGHT, LETTER_WIDTH, 1])
    return fake


def create_sample(generator, label=None, brightness_factor=1.0, is_resized=False):

    if label is None:
        label = [ALL_CHAR_SET[random.randint(0, NUM_CLASSES - 1)] for _ in range(MAX_CAPTCHA)]

    images = [gen_img(x, generator, brightness_factor) for x in label]

    image = np.concatenate(images, axis=1)

    return image, label


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), f"../../SavedModels/{CGAN_MODEL}/Generator/weights.h5")
    generator = load_model(path)
    img, label = create_sample(generator, brightness_factor=1.25)
    img = np.reshape(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img = (img * 255 / np.max(img)).astype('uint8')
    img = Image.fromarray(img, mode='L').resize((100, 40))
    img.save('seq.png')
