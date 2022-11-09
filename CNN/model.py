import os
from itertools import product
from math import ceil
from random import shuffle

import numpy as np
import tensorflow as tf
from keras_preprocessing.image import save_img
from tensorflow.keras import layers

from GAN.utils.one_hot_encoding import encode


class YMLModel:
    def __init__(self, height, width, num_classes, latent_dim, char_set, max_captcha):
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.char_set = char_set
        self.max_captcha = max_captcha

        # TODO maybe padding != same
        self.model = tf.keras.Sequential()

        self.model.add(layers.Input(shape=(self.height, self.width, 1)))
        self.model.add(layers.Conv2D(32, kernel_size=3, padding='same', name='conv1'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))  # drop 50% of the neuron
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPool2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2D(32, kernel_size=3, padding='same', name='conv2'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))  # drop 50% of the neuron
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPool2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2D(64, kernel_size=3, padding='same', name='conv3'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))  # drop 50% of the neuron
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPool2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2D(64, kernel_size=3, padding='same', name='conv4'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.1))  # drop 50% of the neuron
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPool2D(pool_size=(2, 2)))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(1024))
        self.model.add(layers.Dropout(0.1))  # drop 50% of the neuron
        self.model.add(layers.ReLU())

        self.model.add(layers.Dense(self.num_classes, activation='sigmoid'))

    def summary(self):
        return self.model.summary()

    def make_samples(self, batch_size=32, val_percent=0.3):
        samples = [x for x in product(self.char_set, repeat=self.max_captcha)]
        # samples = [x for x in product(self.char_set[11:15], repeat=self.max_captcha)]
        num_samples = len(samples) // batch_size * batch_size
        samples = samples[:num_samples]
        shuffle(samples)

        train_len = ceil(num_samples*(1 - val_percent))

        train_data = samples[:train_len]
        val_data = samples[train_len:]

        return train_data, val_data

    def create_sample_image(self, model, label):
        noise = tf.random.normal(shape=(1, self.latent_dim))

        label = np.reshape(label, (1, 144))
        label = tf.convert_to_tensor(label, np.float32)

        # Combine the noise and the labels and run inference with the generator.
        noise_and_labels = tf.concat([noise, label], 1)
        fake = model.predict(noise_and_labels)
        fake = np.reshape(fake, (40, 100, 1))

        # save_img(os.getcwd() + "/some_test.png", fake)
        return fake

    def generator(self, model, samples, batch_size=32):
        while True:  # Loop forever so the generator never terminates
            # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
            for offset in range(0, len(samples), batch_size):
                # Get the samples you'll use in this batch
                batch_samples = samples[offset:offset + batch_size]

                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []

                # For each example
                for batch_sample in batch_samples:
                    # Load image (X) and label (y)

                    label = encode(batch_sample)
                    img = self.create_sample_image(model, label)

                    # Add example to arrays
                    X_train.append(img)
                    y_train.append(label)

                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
                y_train = np.array(y_train)

                # The generator-y part: yield the next training batch
                yield X_train, y_train

    def compile(self, loss, optimizer, metrics):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, model, batch_size, val_percent, epochs):
        train, val = self.make_samples(batch_size, val_percent=val_percent)

        train_generator = self.generator(model, train, batch_size=batch_size)
        val_generator = self.generator(model, val, batch_size=batch_size)

        self.model.fit(
            train_generator,
            steps_per_epoch=len(train) // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(val) // batch_size,
        )

    def save(self, path='../CNNModels/example.h5'):
        self.model.save(path)
