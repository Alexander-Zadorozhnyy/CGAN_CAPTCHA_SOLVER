# -*- coding: UTF-8 -*-
import numpy as np

import captcha_setting
from captcha_setting import NUM_CLASSES, CNN_CLASSES, ALL_CHAR_SET_LEN, IS_MNIST


def encode(text, is_cnn=False):
    vector = np.zeros(CNN_CLASSES, dtype='float32') if is_cnn \
        else np.zeros(NUM_CLASSES, dtype='float32')

    def char2pos(char):
        k = ord(char) - 48
        if k > 9:
            k = ord(char) - 65 + 10
            if k > 35:
                raise ValueError('error')
        return (k - 1 if k < 24 else k - 2) if not IS_MNIST else k

    for i, char in enumerate(text):
        idx = i * ALL_CHAR_SET_LEN + char2pos(char)
        vector[idx] = 1.0
    return vector


def decode(vec):
    label = ''
    for i in range(vec.shape[1] // captcha_setting.ALL_CHAR_SET_LEN):
        start = captcha_setting.ALL_CHAR_SET_LEN * i
        end = captcha_setting.ALL_CHAR_SET_LEN * (i + 1)
        label += captcha_setting.ALL_CHAR_SET[np.argmax(vec[0, start:end])]

    return label


if __name__ == '__main__':
    e = encode("BK7F", is_cnn=True)
    print(e)
