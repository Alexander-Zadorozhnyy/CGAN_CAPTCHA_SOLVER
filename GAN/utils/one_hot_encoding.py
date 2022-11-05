# -*- coding: UTF-8 -*-
import numpy as np

from GAN.utils.captcha_setting import MAX_CAPTCHA, ALL_CHAR_SET_LEN


def encode(text):
    vector = np.zeros(MAX_CAPTCHA * ALL_CHAR_SET_LEN, dtype='float32')

    def char2pos(c):
        k = ord(c) - 48
        if k > 9:
            c = c.upper()
            k = ord(c) - 65 + 10
            if k > 35:
                k = ord(c) - 97 + 26 + 10
                if k > 61:
                    raise ValueError('error')
        return k

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
    e = encode("BK7H")
    print(decode(e))
