import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from keras_preprocessing.image import save_img

from GAN.utils.utils import load_model, gen_img, create_sample
from captcha_setting import LATENT_DIM, NUM_CLASSES, LETTER_HEIGHT, LETTER_WIDTH, ALL_CHAR_SET, CGAN_MODEL

name = CGAN_MODEL
path = os.path.join(os.getcwd(), f"../../SavedModels/{name}/Generator/weights.h5")
generator = load_model(path)


def save_all():
    # Create some examples
    for i in ALL_CHAR_SET:
        fake = gen_img(i, generator)
        # Saving synthetic image
        save_img(os.getcwd() + f"/{name}_letter_{i}.png", fake)


def create_seq(label=None):
    img, label = create_sample(generator, label)
    img = np.reshape(img, [LETTER_HEIGHT, LETTER_WIDTH * 4])
    img = (img * 255 / np.max(img)).astype('uint8')
    img = Image.fromarray(img, mode='L').resize((100, 40))
    img.save('seq.png')
    print(label)
    with open('seq.txt', 'w') as f:
        f.write(''.join(label))


create_seq('P5D8')
# save_all()
