# -*- coding: UTF-8 -*-
import os


NUMBER = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

NUM_CLASSES = ALL_CHAR_SET_LEN
LETTER_HEIGHT = 40
LETTER_WIDTH = 24
IS_MNIST = False

CGAN_MODEL = 'cluster_19_batch_64_all_0.0003_1000_model'

# MNIST PARAMS
# LETTER_HEIGHT = LETTER_WIDTH = 28
# NUM_CLASSES = 10
# IS_MNIST = True

NUM_CHANNELS = 1
BATCH_SIZE = 64  # Switch to 64?

LATENT_DIM = 100

SAVED_MODEL = ''

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 96
CNN_CLASSES = MAX_CAPTCHA * ALL_CHAR_SET_LEN
CNN_EPOCH = 5
VAL_PERCENT = 0.3