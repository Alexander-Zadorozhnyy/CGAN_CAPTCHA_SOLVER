"""
Title: Conditional GAN
Author: Zadorozhnyy Alexander
Date created: 2022/10/15
Description: Training a GAN conditioned on class labels to generate captchas.
"""

# <----------> #
# ImportBlock  #
# <----------> #

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

import tensorflow as tf

from GAN.models.CGAN import ConditionalGAN
from GAN.models.DatasetHelper import DatasetHelper
from GAN.models.Discriminator import Discriminator
from GAN.models.Generator import Generator
from captcha_setting import NUM_CLASSES, LETTER_HEIGHT, LETTER_WIDTH, LATENT_DIM, NUM_CHANNELS, BATCH_SIZE

# <----------> #
# ImportBlock  #
# <----------> #

"""
## Loading the MNIST dataset and preprocessing it
"""

# MNIST TEST
# We'll use all the available examples from both the training and test
# sets.
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# all_digits = np.concatenate([x_train, x_test])
# all_labels = np.concatenate([y_train, y_test])
#
# all_digits = all_digits.astype("float32") / 255.0
# all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
# all_labels = to_categorical(all_labels, 10)
#
# # Create tf.data.Dataset.
# dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
# dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

# Scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels.

"""
## Prepare dataset
"""

folder = 'data/clusters/cluster_19_single_letters'
letters = None
datasetInitializer = DatasetHelper(folder, LETTER_HEIGHT, LETTER_WIDTH, letters)
dataset = datasetInitializer.create_dataset(batch_size=BATCH_SIZE)

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
## Set training parameters
"""
lr = 0.0003
d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)  #

model_name = f'cluster_19_batch_{BATCH_SIZE}_all_{lr}_2500_model'
saved_model = None  # "../SavedModels/cluster_19_batch_64_all_0.0003_2000_model"

EPOCH = 1000
"""
## Creating the discriminator and generator
"""

generator = Generator(in_channels=generator_in_channels, optimizer=g_optimizer, loss_fn=loss_fn,
                      height=LETTER_HEIGHT,
                      width=LETTER_WIDTH)
discriminator = Discriminator(size=(LETTER_HEIGHT, LETTER_WIDTH), in_channels=discriminator_in_channels,
                              optimizer=d_optimizer, loss_fn=loss_fn)

"""
## Creating or loading a `ConditionalGAN` model
"""

cond_gan = ConditionalGAN(image_size=(LETTER_HEIGHT, LETTER_WIDTH),
                          num_classes=NUM_CLASSES,
                          discriminator=discriminator,
                          generator=generator,
                          latent_dim=LATENT_DIM)

if saved_model is not None:
    cond_gan.load_weights(saved_model)


"""
## Training the Conditional GAN
"""

cond_gan.compile(loss_fn=loss_fn)
cond_gan.fit(dataset, epochs=EPOCH, callbacks=[])

"""
## Save trained model
"""

cond_gan.save_weights(path='../SavedModels', name=model_name)
