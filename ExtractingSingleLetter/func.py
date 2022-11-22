import os

import cv2
from PIL import Image

os.chdir('../GAN/data/clusters')

letter_size = {'1': 18, '2': 23, '3': 20, '4': 21, '5': 23,
               '6': 22, '7': 22, '8': 24, '9': 23, 'A': 25,
               'B': 24, 'C': 18, 'D': 23, 'E': 25, 'F': 26,
               'G': 26, 'H': 23, 'I': 28, 'J': 22, 'K': 20, 'L': 20, 'M': 24, 'N': 23,
               'P': 22, 'Q': 25, 'R': 23, 'S': 22, 'T': 24,
               'U': 28, 'V': 22, 'W': 25, 'X': 23, 'Y': 23, 'Z': 25}


def create_all_dir():
    dirs = os.listdir()
    for x in dirs:
        if '_letters' not in x:
            os.mkdir(x + '_single_letters')

    letters_dirs = [x + '_single_letters' for x in dirs if '_letters' not in x]
    for x in letters_dirs:
        for key in letter_size.keys():
            os.mkdir(f'{x}/{key}')


def crop(catalog, name):
    img = cv2.imread(f'{catalog}/{name}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    label = ''
    with open(f'{catalog}/{name}.txt', 'r') as f:
        label = f.read()

    for i in range(4):
        count = len(os.listdir(f'{catalog}_single_letters/{label[i]}'))
        cv2.imwrite(f"{catalog}_single_letters/{label[i]}/{count}.png", img[:, i*25 - i*10:min(25*i + 35, 100)])
    # indent = 0
    # preper = 8
    # postper = 8
    # for i in range(3):
    #     imgCropped = img[:,
    #                  indent - preper if indent - preper > 0 else indent:
    #                  indent + letter_size[label[i]] + postper if indent + letter_size[label[i]] + postper < 100 else indent + letter_size[label[i]]]
    #     count = len(os.listdir(f'{catalog}_single_letters/{label[i]}'))
    #     imgCropped = cv2.resize(imgCropped, (40, 30), interpolation=cv2.INTER_NEAREST)
    #     indent += letter_size[label[i]]
    #
    # imgCropped = img[:,
    #              100 - letter_size[label[3]] - 8:]
    # count = len(os.listdir(f'{catalog}_single_letters/{label[3]}'))
    # cv2.imwrite(f"{catalog}_single_letters/{label[3]}/{count}.png", imgCropped)


def create_all_crops(filepath):
    files = [x.replace('.png', '') for x in os.listdir(filepath) if '.png' in x]
    for x in files:
        print(x)
        crop(filepath, x)


def resize(filepath):
    for x in os.listdir(filepath):
        files = [f'{filepath}/{x}/{y}' for y in os.listdir(f'{filepath}/{x}')]
        for file in files:
            img = Image.open(file)
            img = img.resize((24, 40))
            img.save(file)
            img.close()


if __name__ == "__main__":
    # create_all_dir()
    for i in range(19, 20):
        # print(i)
        # create_all_crops(f'cluster_{i}')
        # print('created all crooped images')
        resize(f'cluster_{i}_single_letters')
        print('resized all images')