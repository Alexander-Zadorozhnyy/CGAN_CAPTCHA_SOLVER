# -*- coding: UTF-8 -*-
"""
Title: CAPTCHA SOLVER
Author: Zadorozhnyy Alexander
Date created: 2022/12/1
Description: Training a CAPTCHA Solver based on ResNet architecture.
"""
import argparse
import os
import random

import numpy as np
from PIL import Image
import tensorflow as tf

from src.CNN.model import YMLModel
from src.GAN.utils.one_hot_encoding import encode
from captcha_setting import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, \
    CNN_BATCH_SIZE, CNN_EPOCH, CNN_VAL_PERCENT, CNN_CLASSES


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gen_data', type=str,
                        default='generated_data', help="root path to generated captcha's dataset")
    parser.add_argument('--orig_data', type=str,
                        default='original_data', help="root path to original captcha's dataset")
    parser.add_argument('--num_gen_train', type=int,
                        default=50000, help="number of generated captchas to train model")
    parser.add_argument('--num_orig_train', type=int,
                        default=5000, help="number of original captchas to train model")
    parser.add_argument('--num_gen_test', type=int,
                        default=10000, help="number of generated captchas to val model")
    parser.add_argument('--num_orig_test', type=int,
                        default=2000, help="number of original captchas to val model")
    parser.add_argument('--model_name', type=str,
                        default='model', help='root path where to save model')
    parser.add_argument('--saved_model_name', type=str,
                        default=None, help='root path to saved_model')
    return vars(parser.parse_args())


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
        img = Image.open(elem).resize((IMAGE_WIDTH, IMAGE_HEIGHT))\
            .convert("L" if NUM_CHANNELS == 1 else "RGB")

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
        img = Image.open(elem).resize((IMAGE_WIDTH, IMAGE_HEIGHT))\
            .convert("L" if NUM_CHANNELS == 1 else "RGB")

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


def main(gen_data, orig_data, model_name, saved_model_name):
    model = YMLModel(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), num_classes=CNN_CLASSES)
    model.set_resnet_model()

    if saved_model_name is not None:
        model.load(name=saved_model_name)

    model.summary()

    train_data, test_data = get_data(gen_data=gen_data,
                                     orig_data=orig_data)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  opt=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  metrics=[tf.keras.metrics.Accuracy()])

    history = model.fit(data=train_data,
                        epochs=CNN_EPOCH,
                        batch_size=CNN_BATCH_SIZE,
                        val_percent=CNN_VAL_PERCENT,
                        callbacks=[])

    model.evaluate(test_data)

    model.save(name=model_name)

    model.plot_training(history)


if __name__ == '__main__':
    # saved = resnet_44000_000001_30 - best model
    # gen_data = ['generated_data', 40000, 5000]
    # orig_data = ['original_data', 4000, 1000]
    args = get_parser_args()
    main(gen_data=[args['gen_data'], args['num_gen_train'], args['num_gen_test']],
         orig_data=[args['orig_data'], args['num_orig_train'], args['num_orig_test']],
         model_name=args['model_name'],
         saved_model_name=args['saved_model_name'])
