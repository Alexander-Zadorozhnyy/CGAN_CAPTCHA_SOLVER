# -*- coding: UTF-8 -*-
import os
# 验证码中的字符
# string.digits + string.ascii_uppercase


NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
#ALPHABET = []
ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4
NUM_CLASSES = MAX_CAPTCHA * ALL_CHAR_SET_LEN
NUM_CHANNELS = 1
BATCH_SIZE = 32  # Switch to 64
BUFFER_SIZE = 1024

LATENT_DIM = 128

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 100


TRAIN_CLUSTER_DATASET = 'data' + os.path.sep + 'clusters' + os.path.sep + 'cluster_0'
SAVED_MODEL = ''
CGAN_EPOCH = 10000


TRAIN_DATASET_PATH = 'data' + os.path.sep + 'train'
TEST_DATASET_PATH = 'data' + os.path.sep + 'test'
VAL_DATASET_PATH = 'data' + os.path.sep + 'val'