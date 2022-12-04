# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.keras import layers


class Discriminator:
    def __init__(self, size, in_channels, optimizer, loss_fn):
        self.size = size
        self.in_channels = in_channels
        self.d_optimizer = optimizer
        self.loss_fn = loss_fn
        self.d_loss = None
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

        self.model = tf.keras.Sequential(
            [
                layers.InputLayer((self.size[0], self.size[1], self.in_channels)),

                layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same"),
                layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
                layers.LeakyReLU(0.2),

                layers.Conv2D(64 * 2, (4, 4), strides=(3, 3), padding="same"),
                layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
                layers.LeakyReLU(0.2),

                layers.Conv2D(64 * 4, (4, 4), strides=(3, 3), padding="same"),
                layers.BatchNormalization(momentum=0.1, epsilon=0.8, center=1.0, scale=0.02),
                layers.LeakyReLU(0.2),

                layers.Flatten(),
                layers.Dropout(.4),
                layers.Dense(1, activation='sigmoid'),
            ],
            name="discriminator",
        )

    def predict(self, captcha):
        return self.model(captcha)

    def train(self, images, labels):
        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.predict(images)
            self.d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(self.d_loss, self.model.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights)
        )

    def update_weights(self):
        self.disc_loss_tracker.update_state(self.d_loss)

    def get_loss_track(self):
        return self.disc_loss_tracker.result()


if __name__ == "__main__":
    discriminator = Discriminator(size=(40, 24),
                                  in_channels=100,
                                  optimizer=None,
                                  loss_fn=tf.keras.losses.BinaryCrossentropy())
    model_img_file = 'discriminator.png'
    tf.keras.utils.plot_model(discriminator.model, to_file=model_img_file,
                              show_shapes=True,
                              show_layer_activations=True,
                              show_dtype=True,
                              show_layer_names=True)
    print(discriminator.model.summary())
