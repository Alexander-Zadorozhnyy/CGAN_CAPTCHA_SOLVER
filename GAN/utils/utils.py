import os
import random
from itertools import product

import numpy as np
import tensorflow as tf
from PIL import Image

from GAN.models.Generator import Generator
from GAN.utils.one_hot_encoding import encode
from captcha_setting import LATENT_DIM, NUM_CLASSES, LETTER_HEIGHT, LETTER_WIDTH, ALL_CHAR_SET, MAX_CAPTCHA


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


def gen_img(name, generator):
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
    fake = np.reshape(fake, [LETTER_HEIGHT, LETTER_WIDTH, 1])

    return fake


def create_sample(generator, label=None):
    image = None
    if label is None:
        label = [ALL_CHAR_SET[random.randint(0, NUM_CLASSES - 1)] for _ in range(MAX_CAPTCHA)]

    image = np.concatenate(
            [gen_img(x, generator) for x in label],
            axis=1
        )
    return image, label


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), "../../SavedModels/mnist_200_model/Generator/weights.h5")
    generator = load_model(path)
    img, label = create_sample(generator)
    img = np.reshape(img, [LETTER_HEIGHT, LETTER_WIDTH * 4])
    img = (img * 255 / np.max(img)).astype('uint8')
    img = Image.fromarray(img, mode='L').resize((100, 40))
    img.save('seq.png')
