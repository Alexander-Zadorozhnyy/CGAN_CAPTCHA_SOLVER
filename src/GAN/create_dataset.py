import argparse
import os
import random

from captcha_setting import ALL_CHAR_SET, NUM_CLASSES, MAX_CAPTCHA
from src.GAN.tests.test import create_all_seq


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_folder', type=str,
                        default='data', help='root path where to save dataset')
    parser.add_argument('--count', type=int,
                        default=10000, help='number of CAPTCHA to create')
    return vars(parser.parse_args())


def create_random_samples_for_solver(path, count):
    for i in range(count):
        label = [ALL_CHAR_SET[random.randint(0, NUM_CLASSES - 1)] for _ in range(MAX_CAPTCHA)]
        create_all_seq(path=os.path.join(path, "".join(label) + '.png'),
                       label=label,
                       brightness_factor=1.25)
        if i % 5000 == 0:
            print(i)


if __name__ == '__main__':
    args = get_parser_args()
    create_random_samples_for_solver(path=args['dataset_folder'], count=args['count'])
