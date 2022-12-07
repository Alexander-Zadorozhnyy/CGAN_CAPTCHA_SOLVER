# -*- coding: UTF-8 -*-
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import argparse

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model as cnn_lm

from captcha_setting import IMAGE_WIDTH, IMAGE_HEIGHT
from src.GAN.utils.one_hot_encoding import decode


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', type=str,
                        default='app/models/', help='root path to models')
    parser.add_argument('--model', type=str,
                        default='solver.h5', help='name of captcha solver')
    parser.add_argument('--captcha_root', type=str,
                        default='app/example.png', help='root path to captcha')
    return vars(parser.parse_args())


def predict(cnn, name):
    image = Image.open(name).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = image.convert('L')
    image = np.asarray(image)
    image = image.astype("float32") / 255
    image = image.reshape(1, *image.shape)

    predict_label = cnn(image)
    return decode(predict_label)


def main():
    args = get_parser_args()
    path = os.path.join(os.getcwd(), args['data_root'], args['model'])
    cnn = cnn_lm(path)
    print("load YMLModel net.")
    name = os.path.join(os.getcwd(), args['captcha_root'])
    predicted_label = predict(cnn=cnn, name=name)
    print(f'\n--> Predicted captcha label: {predicted_label}')


if __name__ == '__main__':
    main()
