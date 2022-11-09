# -*- coding: UTF-8 -*-
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

import captcha_setting
from CNN.model import YMLModel
from GAN.utils.one_hot_encoding import encode, decode
from captcha_setting import LATENT_DIM, ALL_CHAR_SET


def main():
    def create_sample_image(label):
        noise = tf.random.normal(shape=(1, LATENT_DIM))

        label = np.reshape(label, (1, 144))
        label = tf.convert_to_tensor(label, np.float32)

        # Combine the noise and the labels and run inference with the generator.
        noise_and_labels = tf.concat([noise, label], 1)
        fake = generator.predict(noise_and_labels)
        fake = np.reshape(fake, (1, 40, 100, 1))

        return fake

    # device = torch.device("cuda")
    path = "../CNNModels/example.h5"
    cnn = load_model(path)

    path = "../GeneratorModels/my_model.h5"
    generator = load_model(path)
    print("YML Captcha solver net loaded!")

    # test_dataloader = my_dataset.get_test_train_data_loader()
    correct = 0
    total = 0
    for _ in range(1000):
        label = ''.join(random.sample(ALL_CHAR_SET, 4))
        label_hot_encoding = encode(label)
        image = create_sample_image(label_hot_encoding)

        predict_label = cnn.predict(image)

        c0 = captcha_setting.ALL_CHAR_SET[
            np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN])]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN])]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0,
                                                    2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN])]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0,
                                                    3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN])]

        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        print(label, predict_label)

        # save_image(vimage,'temp_result/'+str(i)+'.png')
        # print(predict_label.upper(),'>>>>>',true_label)
        total += 1
        if (label == predict_label):
            correct += 1
        if (total % 2000 == 0):
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))


if __name__ == '__main__':
    main()
