# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model as cnn_lm

from GAN.utils.one_hot_encoding import decode


def predict(cnn, name):
    image = Image.open(name).resize((96, 40))
    image = image.convert('L')
    image = np.asarray(image)
    image = image.astype("float32") / 255
    image = image.reshape(1, *image.shape)

    predict_label = cnn(image)
    return decode(predict_label)


def main(name):
    path = f"CNNModels/{name}"
    cnn = cnn_lm(path)
    print("load YMLModel net.")
    name = os.path.join(os.getcwd(), 'orig_train.png')
    predicted_label = predict(cnn=cnn, name=name)
    print(predicted_label)


if __name__ == '__main__':
    main('resnet_44000_000001_30/model.h5')  # resnet_44000_000001_30 -- 65%
