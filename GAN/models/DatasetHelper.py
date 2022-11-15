import os

import numpy as np
from PIL import Image
import tensorflow as tf

from captcha_setting import BATCH_SIZE
from GAN.utils.one_hot_encoding import encode


class DatasetHelper:
    def __init__(self, folder, height, width, letters=None):
        self.height = height
        self.width = width

        self.image_paths = []
        self.label_paths = []

        if letters is None:
            for dir in os.listdir(folder):
                    files = os.listdir(folder + '/' + dir)

                    for x in files:
                        self.image_paths += [os.path.join(folder, dir, x)]
                        self.label_paths += [dir]
        else:
            for dir in os.listdir(folder):
                if dir in letters:
                    files = os.listdir(folder + '/' + dir)

                    for x in files:
                        self.image_paths += [os.path.join(folder, dir, x)]
                        self.label_paths += [dir]

        self.image_paths = self.image_paths[:(len(self.image_paths) // BATCH_SIZE) * BATCH_SIZE]
        self.label_paths = self.label_paths[:len(self.image_paths)]
        self.images_shape = None
        self.labels_shape = None

    def get_images_shape(self):
        return self.images_shape

    def get_labels_shape(self):
        return self.labels_shape

    def get_images(self):
        images = np.array([np.array(Image.open(x).convert('L')) for x in self.image_paths]).astype("float32") / 255.0
        images = np.reshape(images, (-1, self.height, self.width, 1))
        self.images_shape = images.shape
        return images

    def get_labels(self):
        labels = []
        for x in self.label_paths:
            labels += [encode(x)]
        labels = np.array(labels)
        self.labels_shape = labels.shape
        return labels

    def create_dataset(self, batch_size=32):
        # Create tf.data.Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((self.get_images(), self.get_labels()))
        dataset = dataset.shuffle(self.__len__()).batch(batch_size)
        return dataset

    def __len__(self):
        return len(self.image_paths)
