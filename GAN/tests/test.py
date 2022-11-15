import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model



from keras_preprocessing.image import save_img

from GAN.models.Generator import Generator
from captcha_setting import LATENT_DIM, NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH
from GAN.utils.one_hot_encoding import encode

path = "../../SavedModels/cluster_10_letters_1-9_model/Generator/weights.h5"

# Params advertisement
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
generator_in_channels = LATENT_DIM + NUM_CLASSES
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Create generator
generator = Generator(in_channels=generator_in_channels, optimizer=g_optimizer, loss_fn=loss_fn,
                      height=IMAGE_HEIGHT,
                      width=IMAGE_WIDTH)

# Load pretrained weights
generator.model.load_weights(os.path.join(os.getcwd(), path))

# Create some examples
for i in range(1, 10):
    # Sample noise for testing.
    noise = tf.random.normal(shape=(1, LATENT_DIM))
    # label = encode("H")
    label = encode(str(i))
    # Prepare label for generator
    label = np.reshape(label, (1, NUM_CLASSES))
    label = tf.convert_to_tensor(label, np.float32)

    # Combine the noise and the labels and run inference with the generator.
    noise_and_labels = tf.concat([noise, label], 1)
    fake = generator.generate(noise_and_labels)
    fake = np.reshape(fake, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # Saving synthetic image
    save_img(os.getcwd() + f"/cluster_1_1-9_all_letter_{i}.png", fake)
