# -*- coding: UTF-8 -*-
"""
Title: CAPTCHA SOLVER
Author: Zadorozhnyy Alexander
Date created: 2022/12/1
Description: Training a CAPTCHA Solver based on ResNet architecture.
"""

import os
import random

import numpy as np
from PIL import Image
import tensorflow as tf

from CNN.model import YMLModel
from GAN.utils.one_hot_encoding import encode
from captcha_setting import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, \
    CNN_BATCH_SIZE, CNN_EPOCH, CNN_VAL_PERCENT, CNN_CLASSES


# some_data = [path, count_train, count_test]
def get_data(gen_data: list, orig_data: list):
    # Load the data and split it between train and test sets
    all_gen_cap = [os.path.join(os.getcwd(), gen_data[0], x) for x in
                   os.listdir(os.path.join(os.getcwd(), gen_data[0]))]
    random.shuffle(all_gen_cap)

    all_orig_cap = [os.path.join(os.getcwd(), orig_data[0], x) for x in
                    os.listdir(os.path.join(os.getcwd(), orig_data[0]))]
    all_orig_cap = (((orig_data[1] + orig_data[2])
                     // len(all_orig_cap) + 1) * all_orig_cap)[:orig_data[1]]

    train = np.asarray(all_gen_cap[:gen_data[1]] + all_orig_cap[:orig_data[1]])
    test = np.asarray(
        all_gen_cap[gen_data[1]:gen_data[1] + gen_data[2]] +
        all_orig_cap[orig_data[1]:orig_data[1] + orig_data[2]])

    np.random.shuffle(train)
    np.random.shuffle(test)

    x_train, y_train, x_test, y_test = [], [], [], []

    # convert labels to binary class vectors and present captchas as matrices
    print('Loading train_data...')
    for i, elem in enumerate(train):
        img = Image.open(elem).resize((IMAGE_WIDTH, IMAGE_HEIGHT)).convert("L")

        x_train += [np.array(img)]
        # os.path.splitdrive()
        y_train += [encode(os.path.split(elem)[-1].replace('.png', ''), is_cnn=True)]

        if i % 5000 == 0:
            print(f'Loaded of train_data: {i}.')

    y_train = np.asarray(y_train)
    # Scale images to the [0, 1] range
    x_train = np.asarray(x_train).astype("float32") / 255
    print('Train data loaded!')

    # convert labels to binary class vectors and present captchas as matrices
    print('Loading test_data...')
    for i, elem in enumerate(test):
        img = Image.open(elem).resize((IMAGE_WIDTH, IMAGE_HEIGHT)).convert("L")

        x_test += [np.array(img)]
        # os.path.splitdrive()
        y_test += [encode(os.path.split(elem)[-1].replace('.png', ''), is_cnn=True)]

        if i % 5000 == 0:
            print(f'Loaded of test_data: {i}.')

    y_test = np.asarray(y_test)
    # Scale images to the [0, 1] range
    x_test = np.asarray(x_test).astype("float32") / 255
    print('Test data loaded!')

    return [x_train, y_train], [x_test, y_test]


def main():
    model = YMLModel(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), num_classes=CNN_CLASSES)
    model.set_resnet_model()

    saved = ''  # resnet_44000_000001_30 - best model

    if saved != '':
        model.load(path='../CNNModels', name=saved)

    model.summary()

    train_data, test_data = get_data(gen_data=['res_data', 10000, 5000],
                                     orig_data=['orig_data', 4000, 1000])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  opt=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  metrics=[tf.keras.metrics.Accuracy()])

    history = model.fit(data=train_data,
                        epochs=CNN_EPOCH,
                        batch_size=CNN_BATCH_SIZE,
                        val_percent=CNN_VAL_PERCENT,
                        callbacks=[])

    model.evaluate(test_data)

    model.save(path='../CNNModels', name='resnet_10000_000001_10')

    model.plot_training(history)


if __name__ == '__main__':
    main()
