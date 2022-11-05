import os

import numpy as np
import tensorflow as tf
from keras.models import load_model


# Sample noise for testing.
from keras_preprocessing.image import save_img

from GAN.utils.captcha_setting import LATENT_DIM
from GAN.utils.one_hot_encoding import encode

path = "my_model.h5"
saved_model = load_model(path)

interpolation_noise = tf.random.normal(shape=(1, LATENT_DIM))
label = encode("BK7H")
label = np.reshape(label, (1, 144))
label = tf.convert_to_tensor(label, np.float32)

# Combine the noise and the labels and run inference with the generator.
noise_and_labels = tf.concat([interpolation_noise, label], 1)
fake = saved_model.predict(noise_and_labels)
fake = np.reshape(fake, (40, 100, 1))

save_img(os.getcwd() + "/some_test.png", fake)
