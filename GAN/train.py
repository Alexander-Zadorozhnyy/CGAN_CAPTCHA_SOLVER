# -*- coding: UTF-8 -*-
"""
Title: Conditional GAN
Author: Zadorozhnyy Alexander
Date created: 2022/10/15
Description: Training a GAN conditioned on class labels to generate captchas.
"""

import tensorflow as tf

from GAN.models.cgan import ConditionalGAN
from GAN.models.dataset_helper import DatasetHelper
from GAN.models.discriminator import Discriminator
from GAN.models.generator import Generator
from captcha_setting import LETTER_HEIGHT, LETTER_WIDTH, CGAN_LATENT_DIM, \
    NUM_CLASSES, NUM_CHANNELS, CGAN_BATCH_SIZE, CLUSTER, CGAN_LR_D, CGAN_LR_G, CGAN_EPOCH

# MNIST TEST
# import numpy as np
# from keras.datasets import mnist
# from keras.utils import to_categorical

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
# dataset = dataset.shuffle(buffer_size=1024).batch(CGAN_BATCH_SIZE)

# Scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels.

# <-------------> #
# Prepare dataset #
# <-------------> #

folder = f'data/clusters/cluster_{CLUSTER}_single_letters'
LETTERS = None  # letters = ['1', '2' ...]
datasetInitializer = DatasetHelper(folder, LETTER_HEIGHT, LETTER_WIDTH, LETTERS)
dataset = datasetInitializer.create_dataset(batch_size=CGAN_BATCH_SIZE)

print(f"Shape of training images: {datasetInitializer.get_images_shape()}")
print(f"Shape of training labels: {datasetInitializer.get_labels_shape()}")

# <-------------> #
# Prepare dataset #
# <-------------> #

# <-------------------------------------------------------------------------> #
# Calculating the number of input channel for the generator and discriminator #
# <-------------------------------------------------------------------------> #

GENERATOR_INPUT_CHANNELS = CGAN_LATENT_DIM + NUM_CLASSES
DISCRIMINATOR_INPUT_CHANNELS = NUM_CHANNELS + NUM_CLASSES
print(GENERATOR_INPUT_CHANNELS, DISCRIMINATOR_INPUT_CHANNELS)

# <-------------------------------------------------------------------------> #
# Calculating the number of input channel for the generator and discriminator #
# <-------------------------------------------------------------------------> #

# <---------------------> #
# Set training parameters #
# <---------------------> #

d_optimizer = tf.keras.optimizers.Adam(learning_rate=CGAN_LR_D)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=CGAN_LR_G)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

model_name = f'cluster_{CLUSTER}_batch_{CGAN_BATCH_SIZE}_all_{CGAN_EPOCH}_model'
SAVED_MODEL = None  # "../SavedModels/cluster_19_batch_64_all_0.0003_2000_model"

# <---------------------> #
# Set training parameters #
# <---------------------> #

# <--------------------------------------> #
# Creating the discriminator and generator #
# <--------------------------------------> #

generator = Generator(in_channels=GENERATOR_INPUT_CHANNELS,
                      optimizer=g_optimizer,
                      loss_fn=loss_fn,
                      height=LETTER_HEIGHT,
                      width=LETTER_WIDTH)
discriminator = Discriminator(size=(LETTER_HEIGHT, LETTER_WIDTH),
                              in_channels=DISCRIMINATOR_INPUT_CHANNELS,
                              optimizer=d_optimizer,
                              loss_fn=loss_fn)

# <--------------------------------------> #
# Creating the discriminator and generator #
# <--------------------------------------> #

# <------------------------------------------> #
# Creating or loading a `ConditionalGAN` model #
# <------------------------------------------> #

cond_gan = ConditionalGAN(image_size=(LETTER_HEIGHT, LETTER_WIDTH),
                          num_classes=NUM_CLASSES,
                          discriminator=discriminator,
                          generator=generator,
                          latent_dim=CGAN_LATENT_DIM)

if SAVED_MODEL is not None:
    cond_gan.load_weights(SAVED_MODEL)

# <------------------------------------------> #
# Creating or loading a `ConditionalGAN` model #
# <------------------------------------------> #

# <--------------------------> #
# Training the Conditional GAN #
# <--------------------------> #

cond_gan.compile(loss_fn=loss_fn)
cond_gan.fit(dataset, epochs=CGAN_EPOCH, callbacks=[])

# <--------------------------> #
# Training the Conditional GAN #
# <--------------------------> #

# <----------------> #
# Save trained model #
# <----------------> #

cond_gan.save_weights(path='../SavedModels', name=model_name)

# <----------------> #
# Save trained model #
# <----------------> #
