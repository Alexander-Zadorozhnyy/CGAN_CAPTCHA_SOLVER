# Create the discriminator.
import tensorflow as tf
from tensorflow.keras import layers


class Discriminator:
    def __init__(self, size, in_channels, optimizer, loss_fn, saved_model=None):
        self.size = size
        self.in_channels = in_channels
        self.d_optimizer = optimizer
        self.loss_fn = loss_fn
        self.d_loss = None
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

        self.model = None

        if self.model is not None:
            self.model = saved_model
        else:
            self.model = tf.keras.Sequential(
                [
                    layers.InputLayer((self.size[0], self.size[1], self.in_channels)),
                    layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                    layers.LeakyReLU(alpha=0.2),
                    layers.Dropout(.2),
                    layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                    layers.LeakyReLU(alpha=0.2),
                    layers.Dropout(.2),
                    layers.GlobalMaxPooling2D(),
                    layers.Dense(1),
                ],
                name="discriminator",
            )

        self.model_ = tf.keras.Sequential(
            [
                layers.InputLayer((self.size[0], self.size[1], self.in_channels)),

                layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(0.2),
                layers.Dropout(0.5),

                layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same"),
                layers.BatchNormalization(),
                layers.LeakyReLU(0.2),
                layers.Dropout(0.5),

                layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                layers.BatchNormalization(),
                layers.LeakyReLU(0.2),
                layers.Dropout(0.5),

                layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same"),
                layers.BatchNormalization(),
                layers.LeakyReLU(0.2),
                layers.Dropout(0.5),

                layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
                layers.BatchNormalization(),
                layers.LeakyReLU(0.2),
                layers.Dropout(0.5),

                layers.Conv2D(512, (3, 3), strides=(1, 1), padding="same"),
                layers.BatchNormalization(),
                layers.LeakyReLU(0.2),
                layers.Dropout(0.5),

                layers.Flatten(),
                layers.Dense(1, activation='sigmoid'),
            ],
            name="discriminator",
        )

    def predict(self, x):
        return self.model(x)

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


# Create the discriminator.
# discriminator_adv = tf.keras.Sequential(
#     [
#         layers.InputLayer((40, 100, discriminator_in_channels)),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same"),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Conv2D(256, (5, 5), padding="same"),
#         layers.Flatten(),
#         # layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
#         # layers.BatchNormalization(),
#         # layers.LeakyReLU(alpha=0.2),
#         # layers.Dropout(0.2),
#         # layers.GlobalMaxPooling2D(),
#         layers.Dense(1, activation="sigmoid"),
#     ],
#     name="discriminator_adv",
# )
# https://blog.paperspace.com/conditional-generative-adversarial-networks/
# https://iq.opengenus.org/face-aging-cgan-keras/
# https://towardsdatascience.com/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8

if __name__ == "__main__":
    discriminator = Discriminator(size=(40, 24),
                                  in_channels=100,
                                  optimizer=None,
                                  loss_fn=tf.keras.losses.BinaryCrossentropy())
    print(discriminator.model.summary())
