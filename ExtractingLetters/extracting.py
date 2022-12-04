# -*- coding: UTF-8 -*-
import os

import cv2
from PIL import Image

from captcha_setting import CLUSTER_NUMBER, LETTER_WIDTH, LETTER_HEIGHT, MAX_CAPTCHA, IMAGE_WIDTH

os.chdir('../GAN/data/clusters')

letter_size = {'1': 18, '2': 23, '3': 20, '4': 21, '5': 23,
               '6': 22, '7': 22, '8': 24, '9': 23, 'A': 25,
               'B': 24, 'C': 18, 'D': 23, 'E': 25, 'F': 26,
               'G': 26, 'H': 23, 'I': 28, 'J': 22, 'K': 20, 'L': 20, 'M': 24, 'N': 23,
               'P': 22, 'Q': 25, 'R': 23, 'S': 22, 'T': 24,
               'U': 28, 'V': 22, 'W': 25, 'X': 23, 'Y': 23, 'Z': 25}


def create_all_dir():
    dirs = os.listdir()
    for directory in dirs:
        if '_letters' not in directory:
            os.mkdir(directory + '_single_letters')

    letters_dirs = [x + '_single_letters' for x in dirs if '_letters' not in x]
    for letter_dir in letters_dirs:
        for key in letter_size:
            os.mkdir(f'{letter_dir}/{key}')


def crop(catalog, name):
    img = cv2.imread(f'{catalog}/{name}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    label = ''
    with open(f'{catalog}/{name}.txt', 'r') as file:
        label = file.read()

    for ind in range(MAX_CAPTCHA):
        count = len(os.listdir(f'{catalog}_single_letters/{label[ind]}'))
        start = ind * LETTER_WIDTH - ind * 10
        end = min(LETTER_WIDTH * ind + 35, IMAGE_WIDTH)
        cv2.imwrite(f"{catalog}_single_letters/{label[ind]}/{count}.png", img[:, start:end])


def create_all_crops(filepath):
    files = [x.replace('.png', '') for x in os.listdir(filepath) if '.png' in x]
    for file in files:
        # print(file)
        crop(filepath, file)


def resize(filepath):
    for directory in os.listdir(filepath):
        files = [f'{filepath}/{directory}/{letter_dir}' for letter_dir in
                 os.listdir(f'{filepath}/{directory}')]
        for file in files:
            img = Image.open(file).resize((LETTER_WIDTH, LETTER_HEIGHT))
            img.save(file)


if __name__ == "__main__":
    # create_all_dir()
    for i in range(CLUSTER_NUMBER):
        # print(i)
        create_all_crops(f'cluster_{i}')
        print('created all crooped images')
        resize(f'cluster_{i}_single_letters')
        print('resized all images')
