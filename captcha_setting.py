# -*- coding: UTF-8 -*-
import os

NUMBER = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ALL_CHAR_SET = NUMBER  # + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 1
NUM_CLASSES = MAX_CAPTCHA * ALL_CHAR_SET_LEN

NUM_CHANNELS = 1
BATCH_SIZE = 32  # Switch to 64?

LATENT_DIM = 100

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 24

# MNIST PARAMS
# IMAGE_HEIGHT = IMAGE_WIDTH = 28
# NUM_CLASSES = 10


TRAIN_CLUSTER_DATASET = 'data' + os.path.sep + 'clusters' + os.path.sep + 'cluster_0'
SAVED_MODEL = ''
CGAN_EPOCH = 10000
CNN_EPOCH = 5
VAL_PERCENT = 0.3


TRAIN_DATASET_PATH = 'data' + os.path.sep + 'train'
TEST_DATASET_PATH = 'data' + os.path.sep + 'test'
VAL_DATASET_PATH = 'data' + os.path.sep + 'val'