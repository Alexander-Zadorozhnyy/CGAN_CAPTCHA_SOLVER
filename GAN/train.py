"""
Title: Conditional GAN
Author: Zadorozhnyy Alexander
Date created: 2022/10/15
Description: Training a GAN conditioned on class labels to generate captchas.
"""

# <----------> #
# ImportBlock  #
# <----------> #

import tensorflow as tf
from tensorflow import keras

from GAN.models.CGAN import ConditionalGAN
from GAN.models.DatasetHelper import DatasetHelper
from GAN.models.Discriminator import Discriminator
from GAN.models.Generator import Generator
from GAN.utils.captcha_setting import NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, LATENT_DIM, NUM_CHANNELS, BUFFER_SIZE, \
    BATCH_SIZE

# <----------> #
# ImportBlock  #
# <----------> #

"""
## Constants and hyperparameters
"""

"""
## Loading the MNIST dataset and preprocessing it
"""

# We'll use all the available examples from both the training and test
# sets.
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# all_digits = np.concatenate([x_train, x_test])
# all_labels = np.concatenate([y_train, y_test])

# Scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels.

folder = 'data/clusters/cluster_0'
datasetInitializer = DatasetHelper(folder, IMAGE_HEIGHT, IMAGE_WIDTH)
dataset = datasetInitializer.create_dataset(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)

print(f"Shape of training images: {datasetInitializer.get_images_shape()}")
print(f"Shape of training labels: {datasetInitializer.get_labels_shape()}")

"""
## Calculating the number of input channel for the generator and discriminator

In a regular (unconditional) GAN, we start by sampling noise (of some fixed
dimension) from a normal distribution. In our case, we also need to account
for the class labels. We will have to add the number of classes to
the input channels of the generator (noise input) as well as the discriminator
(generated image input).
"""

generator_in_channels = LATENT_DIM + NUM_CLASSES
discriminator_in_channels = NUM_CHANNELS + NUM_CLASSES
print(generator_in_channels, discriminator_in_channels)

"""
## Creating the discriminator and generator
"""

d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = Generator(in_channels=generator_in_channels, optimizer=g_optimizer, loss_fn=loss_fn, height=IMAGE_HEIGHT,
                      width=IMAGE_WIDTH)
discriminator = Discriminator(in_channels=discriminator_in_channels, optimizer=d_optimizer, loss_fn=loss_fn)

"""
## Creating a `ConditionalGAN` model
"""

cond_gan = ConditionalGAN(image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                          num_classes=NUM_CLASSES,
                          discriminator=discriminator,
                          generator=generator,
                          latent_dim=LATENT_DIM)


"""
## Training the Conditional GAN
"""

cond_gan.compile()

cond_gan.fit(dataset, epochs=100)

"""
## Interpolating between classes with the trained generator
"""

# We first extract the trained generator from our Conditiona GAN.
trained_gen = cond_gan.generator.model

trained_gen.save('tests/my_model.h5')