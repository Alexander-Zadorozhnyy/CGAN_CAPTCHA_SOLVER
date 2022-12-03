import os
import pickle

import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model, layers

from captcha_setting import IMAGE_HEIGHT, IMAGE_WIDTH, CNN_CLASSES, LATENT_DIM, ALL_CHAR_SET, MAX_CAPTCHA, NUM_CHANNELS


class YMLModel:
    def __init__(self, height, width, num_classes, num_channels):
        self.model = None
        self.opt = None
        self.loss = None
        self.metrics = None

        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.num_classes = num_classes

    def set_resnet_model(self):
        def conv_bn_rl(x, f, k=1, s=1, p='same'):
            x = layers.Conv2D(f, k, strides=s, padding=p)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            return x

        def identity_block(tensor, f):
            x = conv_bn_rl(tensor, f)
            x = conv_bn_rl(x, f, 3)
            x = layers.Conv2D(4 * f, 1)(x)
            x = layers.BatchNormalization()(x)

            x = layers.Add()([x, tensor])
            output = layers.ReLU()(x)
            return output

        def conv_block(tensor, f, s):
            x = conv_bn_rl(tensor, f)
            x = conv_bn_rl(x, f, 3, s)
            x = layers.Conv2D(4 * f, 1)(x)
            x = layers.BatchNormalization()(x)

            shortcut = layers.Conv2D(4 * f, 1, strides=s)(tensor)
            shortcut = layers.BatchNormalization()(shortcut)

            x = layers.Add()([x, shortcut])
            output = layers.ReLU()(x)
            return output

        def resnet_block(x, f, r, s=2):
            x = conv_block(x, f, s)
            for _ in range(r - 1):
                x = identity_block(x, f)
            return x

        input = layers.Input((self.height, self.width, self.num_channels))

        x = conv_bn_rl(input, 64, 7, 2)
        x = layers.MaxPool2D(3, strides=2, padding='same')(x)

        x = resnet_block(x, 64, 3, 1)
        x = resnet_block(x, 128, 4)
        x = resnet_block(x, 256, 6)
        x = resnet_block(x, 512, 3)

        x = layers.GlobalAvgPool2D()(x)
        # x = layers.Dropout(.5)(x)
        output = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(input, output)

    def summary(self):
        print(self.model.summary())

    def predict(self, x):
        return self.model(x)

    def compile(self, opt, loss, metrics):
        self.opt = opt
        self.loss = loss
        self.metrics = metrics

        self.model.compile(optimizer=self.opt,
                           loss=self.loss,
                           metrics=self.metrics)

    def fit(self, data, epochs, batch_size=32, val_percent=0.1, callbacks=[]):
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
        import keras.backend as K

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
    solver = YMLModel(IMAGE_HEIGHT, IMAGE_WIDTH, CNN_CLASSES, NUM_CHANNELS)
    solver.set_resnet_model()
    solver.summary()
