# -*- coding: UTF-8 -*-

# <--- CLUSTERING ---> #
CLUSTER_NUMBER = 30

# <--- CAPTCHA PARAMETERS ---> #
NUMBERS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E',
            'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'P',
            'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
ALL_CHAR_SET = NUMBERS + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

# <--- LETTER PARAMETERS ---> #
NUM_CLASSES = ALL_CHAR_SET_LEN
LETTER_HEIGHT = 40
LETTER_WIDTH = 24
IS_MNIST = False

# <--- IMAGE PARAMETERS ---> #
IMAGE_HEIGHT = 40
IMAGE_WIDTH = 96
NUM_CHANNELS = 1

# <--- CGAN PARAMETERS ---> #
CLUSTER = 19
CGAN_BATCH_SIZE = 64
CGAN_LATENT_DIM = 100
CGAN_EPOCH = 1000
CGAN_LR_G = 0.0003
CGAN_LR_D = 0.0003

# <--- CNN PARAMETERS ---> #
CNN_EPOCH = 25
CNN_BATCH_SIZE = 64
CNN_VAL_PERCENT = 0.1
CNN_CLASSES = MAX_CAPTCHA * ALL_CHAR_SET_LEN

# <--- TRAINED MODELS ---> #
CGAN_MODEL = 'cluster_19_batch_64_all_0.0003_2000_model'
CNN_MODEL = 'resnet_44000_000001_30/model.h5'  # acc = 65%
