# -*- coding: UTF-8 -*-
import numpy as np

from captcha_setting import NUM_CLASSES, CNN_CLASSES, ALL_CHAR_SET_LEN, IS_MNIST


def encode(text, is_cnn=False):
    vector = np.zeros(CNN_CLASSES, dtype='float32') if is_cnn else np.zeros(NUM_CLASSES, dtype='float32')

    def char2pos(c):
        c = c.upper()
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 65 + 10
            if k > 35:
                raise ValueError('error')
        return (k - 1 if k < 24 else k - 2) if not IS_MNIST else k

    for i, c in enumerate(text):
        idx = i * ALL_CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1.0
    return vector


def decode(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % 36
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


if __name__ == '__main__':
    e = encode("BK7F")
    print(decode(e))
