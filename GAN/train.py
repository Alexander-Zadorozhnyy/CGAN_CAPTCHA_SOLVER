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
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical

from GAN.models.CGAN import ConditionalGAN
from GAN.models.DatasetHelper import DatasetHelper
from GAN.models.Discriminator import Discriminator
from GAN.models.Generator import Generator
from captcha_setting import NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, LATENT_DIM, NUM_CHANNELS, BATCH_SIZE

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

folder = 'data/clusters/cluster_10_letters'
letters = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
datasetInitializer = DatasetHelper(folder, IMAGE_HEIGHT, IMAGE_WIDTH, letters)
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

d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

model_name = 'mnist_200_model'
saved_model = None
EPOCH = 30

"""
## Creating the discriminator and generator
"""

generator = Generator(in_channels=generator_in_channels, optimizer=g_optimizer, loss_fn=loss_fn,
                      height=IMAGE_HEIGHT,
                      width=IMAGE_WIDTH)
discriminator = Discriminator(size=(IMAGE_HEIGHT, IMAGE_WIDTH), in_channels=discriminator_in_channels,
                              optimizer=d_optimizer, loss_fn=loss_fn)


"""
## Creating or loading a `ConditionalGAN` model
"""

cond_gan = ConditionalGAN(image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                          num_classes=NUM_CLASSES,
                          discriminator=discriminator,
                          generator=generator,
                          latent_dim=LATENT_DIM)

if saved_model is not None:
    cond_gan.load_weights(saved_model)

# Change this to `model_dir` when not using the downloaded weights
# weights_dir = "./tmp"
#
# latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
# cond_gan.load_weights(latest_checkpoint)
# checkpoint_dir = '/checkpoints'
# checkpoint = tf.train.Checkpoint(d_optimizer=d_optimizer,
#                                  g_optimizer=g_optimizer,
#                                  generator=cond_gan.generator.model,
#                                  discriminator=cond_gan.discriminator.model)
#
# ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath='./tmp/checkpoint',
#     save_weights_only=True,
#     monitor='g_loss',
#     mode='min',
#     save_best_only=True)

"""
## Training the Conditional GAN
"""

cond_gan.compile(loss_fn=loss_fn)
cond_gan.fit(dataset, epochs=EPOCH, callbacks=[])

"""
## Save trained model
"""

cond_gan.save_weights(path='../SavedModels', name=model_name)
