# -*- coding: UTF-8 -*-
"""
Title: Conditional GAN
Author: Zadorozhnyy Alexander
Date created: 2022/10/15
Description: Training a GAN conditioned on class labels to generate captchas.
"""
import argparse

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical

from src.GAN.models.cgan import ConditionalGAN
from src.GAN.models.dataset_helper import DatasetHelper
from src.GAN.models.discriminator import Discriminator
from src.GAN.models.generator import Generator
from captcha_setting import LETTER_HEIGHT, LETTER_WIDTH, CGAN_LATENT_DIM, \
    NUM_CLASSES, NUM_CHANNELS, CGAN_BATCH_SIZE, CGAN_LR_D, CGAN_LR_G, CGAN_EPOCH


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_folder', type=str,
                        default='data', help='root path to dataset')
    parser.add_argument('--symbols', type=list,
                        default='data', help='list of symbols which contain captcha')
    parser.add_argument('--model_name', type=str,
                        default='model', help='root path where to save model')
    parser.add_argument('--saved_model_name', type=str,
                        default=None, help='root path to saved_model')
    return vars(parser.parse_args())


def get_mnist_test_dataset():
    # We'll use all the available examples from both the training and test
    # sets.

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_labels = np.concatenate([y_train, y_test])

    all_digits = all_digits.astype("float32") / 255.0
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    all_labels = to_categorical(all_labels, 10)

    # Create tf.data.Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(CGAN_BATCH_SIZE)


def main(dataset_folder, symbols, model_name, saved_model_name):
    # <-------------> #
    # Prepare dataset #
    # <-------------> #
    dataset_initializer = DatasetHelper(dataset_folder, LETTER_HEIGHT, LETTER_WIDTH, symbols)
    dataset = dataset_initializer.create_dataset(batch_size=CGAN_BATCH_SIZE)

    print(f"Shape of training images: {dataset_initializer.get_images_shape()}")
    print(f"Shape of training labels: {dataset_initializer.get_labels_shape()}")

    # <-------------> #
    # Prepare dataset #
    # <-------------> #

    # <-------------------------------------------------------------------------> #
    # Calculating the number of input channel for the generator and discriminator #
    # <-------------------------------------------------------------------------> #

    generator_input_channels = CGAN_LATENT_DIM + NUM_CLASSES
    discriminator_input_channels = NUM_CHANNELS + NUM_CLASSES
    print(generator_input_channels, discriminator_input_channels)

    # <-------------------------------------------------------------------------> #
    # Calculating the number of input channel for the generator and discriminator #
    # <-------------------------------------------------------------------------> #

    # <---------------------> #
    # Set training parameters #
    # <---------------------> #

    d_optimizer = tf.keras.optimizers.Adam(learning_rate=CGAN_LR_D)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=CGAN_LR_G)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # <---------------------> #
    # Set training parameters #
    # <---------------------> #

    # <--------------------------------------> #
    # Creating the discriminator and generator #
    # <--------------------------------------> #

    generator = Generator(in_channels=generator_input_channels,
                          optimizer=g_optimizer,
                          loss_fn=loss_fn,
                          height=LETTER_HEIGHT,
                          width=LETTER_WIDTH)
    discriminator = Discriminator(size=(LETTER_HEIGHT, LETTER_WIDTH),
                                  in_channels=discriminator_input_channels,
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

    if saved_model_name is not None:
        cond_gan.load_weights(saved_model_name)

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

    cond_gan.save_weights(path='trained_models', name=model_name)

    # <----------------> #
    # Save trained model #
    # <----------------> #


if __name__ == '__main__':
    # folder = f'data/clusters/cluster_{CLUSTER}_single_letters'
    # model_name = f'cluster_{CLUSTER}_batch_{CGAN_BATCH_SIZE}_all_{CGAN_EPOCH}_model'
    # SAVED_MODEL = "../SavedModels/cluster_19_batch_64_all_0.0003_2000_model"
    args = get_parser_args()
    main(dataset_folder=args['dataset_folder'],
         symbols=args['symbols'],
         model_name=args['model_name'],
         saved_model_name=args['saved_model_name'])
