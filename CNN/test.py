# -*- coding: UTF-8 -*-
import os
import random
import time

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model as cnn_lm

from GAN.utils.one_hot_encoding import decode
from GAN.utils.utils import load_model as gan_lm, create_sample
from captcha_setting import ALL_CHAR_SET, CGAN_MODEL, CNN_MODEL


def predict(cnn, name):
    image = Image.open(name).resize((96, 40))
    image = image.convert('L')
    image = np.asarray(image)
    image = image.astype("float32") / 255
    image = image.reshape(1, *image.shape)

    predict_label = cnn(image)
    return decode(predict_label)


def test_gen_cap(gen, cnn, count):
    # Test solver on generated CAPTCHA
    correct = 0
    total = 0
    for _ in range(count):
        label = ''.join(random.sample(ALL_CHAR_SET, 4))
        image, label = create_sample(generator=gen,
                                     label=label,
                                     brightness_factor=1.25,
                                     is_res=True)
        image = image.reshape(1, *image.shape)
        predict_label = cnn(image)

        predict_label = decode(predict_label)

        total += 1
        if label == predict_label:
            correct += 1
        if total % 500 == 0:
            print(f'Test Accuracy of the model on the {total} '
                  f'generated test images: {100 * correct / total}%')
    print(f'Test Accuracy of the model on the {total} '
          f'generated test images: {100 * correct / total}%')


def test_not_trained_cap(cnn, path):
    files = os.listdir(path)
    files = [x.replace('.png', '') for x in files if '.png' in x][:5000]

    total = len(files)
    correct = 0
    for label in files:
        predict_label = predict(cnn=cnn, name=os.path.join(path, label + '.png'))
        if label == predict_label:
            correct += 1
        # print(label, predict_label)

    print(f'Test Accuracy of the model on the {total} not '
          f'trained images: {100 * correct / total}%')


def main():
    path = f"../CNNModels/{CNN_MODEL}"
    cnn = cnn_lm(path)

    path = os.path.join(os.getcwd(), f"../SavedModels/{CGAN_MODEL}/Generator/weights.h5")
    gen = gan_lm(path)
    print("YML Captcha solver net loaded!")

    start_time = time.time()
    test_gen_cap(gen, cnn, 100)
    print(f"--- Generation + predictions on "
          f"random CAPTCHA: {time.time() - start_time} seconds ---")

    start_time = time.time()
    test_not_trained_cap(cnn, path=os.path.join(os.getcwd(), 'res_data'))
    print(f"--- Generated CAPTCHA only for "
          f"speed test: {time.time() - start_time} seconds ---"

    start_time = time.time()
    test_not_trained_cap(cnn, path=os.path.join(os.getcwd(), 'not_trained'))
    print(f"--- Not trained CAPTCHA: {time.time() - start_time} seconds ---")


if __name__ == '__main__':
    main()
