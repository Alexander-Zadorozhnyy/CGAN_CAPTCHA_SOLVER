import os

import numpy as np
from PIL import Image
import tensorflow as tf

from GAN.utils.captcha_setting import BATCH_SIZE
from GAN.utils.one_hot_encoding import encode


class DatasetHelper:
    def __init__(self, folder, height, width):
        self.height = height
        self.width = width

        self.image_paths = [os.path.join(folder, file) for file in os.listdir(folder) if '.png' in file]
        self.image_paths = self.image_paths[:(len(self.image_paths) // BATCH_SIZE) * BATCH_SIZE]

        self.labels_path = [x.replace('.png', '.txt', 1) for x in self.image_paths]

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
        for x in self.labels_path:
            with open(x, 'r') as f:
                labels += [encode(f.read())]
        labels = np.array(labels)
        self.labels_shape = labels.shape
        return labels

    def create_dataset(self, buffer_size, batch_size=1024):
        # Create tf.data.Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((self.get_images(), self.get_labels()))
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
        return dataset

    def __len__(self):
        return len(self.image_paths)