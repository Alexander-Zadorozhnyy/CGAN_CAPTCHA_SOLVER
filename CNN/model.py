# -*- coding: UTF-8 -*-
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend as K
from tensorflow.keras import Model, layers

from captcha_setting import IMAGE_HEIGHT, IMAGE_WIDTH, CNN_CLASSES, NUM_CHANNELS


class YMLModel:
    def __init__(self, shape, num_classes):
        self.model = None
        self.opt = None
        self.loss = None
        self.metrics = None

        self.shape = shape
        self.num_classes = num_classes

    def set_resnet_model(self):
        def conv_bn_rl(tensor, filters, kernel=1, strides=1, p='same'):
            tensor = layers.Conv2D(filters, kernel, strides=strides, padding=p)(tensor)
            tensor = layers.BatchNormalization()(tensor)
            tensor = layers.ReLU()(tensor)
            return tensor

        def identity_block(input_layer, filters):
            tensor = conv_bn_rl(input_layer, filters)
            tensor = conv_bn_rl(tensor, filters, 3)
            tensor = layers.Conv2D(4 * filters, 1)(tensor)
            tensor = layers.BatchNormalization()(tensor)

            tensor = layers.Add()([tensor, input_layer])
            output = layers.ReLU()(tensor)
            return output

        def conv_block(input_layer, filters, strides):
            tensor = conv_bn_rl(input_layer, filters)
            tensor = conv_bn_rl(tensor, filters, 3, strides)
            tensor = layers.Conv2D(4 * filters, 1)(tensor)
            tensor = layers.BatchNormalization()(tensor)

            shortcut = layers.Conv2D(4 * filters, 1, strides=strides)(input_layer)
            shortcut = layers.BatchNormalization()(shortcut)

            tensor = layers.Add()([tensor, shortcut])
            output = layers.ReLU()(tensor)
            return output

        def resnet_block(input_layer, filters, repeat, strides=2):
            tensor = conv_block(input_layer, filters, strides)
            for _ in range(repeat - 1):
                tensor = identity_block(tensor, filters)
            return tensor

        input_layer = layers.Input(self.shape)

        output = conv_bn_rl(input_layer, 64, 7, 2)
        output = layers.MaxPool2D(3, strides=2, padding='same')(output)

        output = resnet_block(output, 64, 3, 1)
        output = resnet_block(output, 128, 4)
        output = resnet_block(output, 256, 6)
        output = resnet_block(output, 512, 3)

        output = layers.GlobalAvgPool2D()(output)
        # output = layers.Dropout(.5)(output)
        output = layers.Dense(self.num_classes, activation='softmax')(output)

        self.model = Model(input_layer, output)

    def summary(self):
        print(self.model.summary())

    def predict(self, captcha):
        return self.model(captcha)

    def compile(self, opt, loss, metrics):
        self.opt = opt
        self.loss = loss
        self.metrics = metrics

        self.model.compile(optimizer=self.opt,
                           loss=self.loss,
                           metrics=self.metrics)

    def fit(self, data, epochs, batch_size=32, val_percent=0.1, callbacks=None):
        history = self.model.fit(data[0], data[1],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=val_percent, callbacks=callbacks)

        return history

    def evaluate(self, data, verbose=0):
        score = self.model.evaluate(data[0], data[1], verbose=verbose)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def save(self, path='../CNNModels', name='example.h5'):
        if not os.path.isdir(os.path.join(os.getcwd(), path)):
            os.makedirs(os.path.join(os.getcwd(), path))
        os.makedirs(os.path.join(os.getcwd(), path, name), exist_ok=True)

        self.model.save_weights(os.path.join(os.getcwd(), path, name, 'weights.h5'))
        self.model.save(os.path.join(os.getcwd(), path, name, 'model.h5'))
        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open(os.path.join(os.getcwd(), path, name, 'optimizer.pkl'), 'wb') as f:
            pickle.dump(weight_values, f)

    def load(self, name, path='../CNNModels'):
        self.model.load_weights(os.path.join(os.getcwd(), path, name, 'weights.h5'))
        # self.generator.model.make_train_function()
        with open(os.path.join(os.getcwd(), path, name, 'optimizer.pkl'), 'rb') as f:
            weight_values = pickle.load(f)

        zero_grads = [tf.zeros_like(w) for w in self.model.trainable_weights]
        self.opt.apply_gradients(zip(zero_grads, self.model.trainable_weights))
        self.opt.set_weights(weight_values)

    @staticmethod
    def plot_training(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        print("run to here")
        print(matplotlib.get_backend())

        plt.plot(epochs, acc, 'r.', label='train_acc')
        plt.plot(epochs, val_acc, 'b', label='val_acc')
        plt.title("Training and validation accuracy")
        plt.legend(loc=0, ncol=2)
        plt.savefig('./accuracy.png')

        plt.figure()
        plt.plot(epochs, loss, 'r.', label='train_loss')
        plt.plot(epochs, val_loss, 'b-', label='val_loss')
        plt.title("Training and validation loss")
        plt.legend(loc=0, ncol=2)
        plt.savefig('./loss.png')
        plt.show()


if __name__ == '__main__':
    solver = YMLModel((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), CNN_CLASSES)
    solver.set_resnet_model()
    solver.summary()
