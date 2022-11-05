# Create the discriminator.
import tensorflow as tf
from tensorflow.keras import layers


class Discriminator:
    def __init__(self, in_channels, optimizer, loss_fn):
        self.in_channels = in_channels
        self.d_optimizer = optimizer
        self.loss_fn = loss_fn
        self.d_loss = None
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

        self.model = tf.keras.Sequential(
            [
                layers.InputLayer((40, 100, self.in_channels)),
                layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.GlobalMaxPooling2D(),
                layers.Dense(1),
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
    discriminator = Discriminator(in_channels=100,
                                  optimizer=None,
                                  loss_fn=tf.keras.losses.BinaryCrossentropy())
    print(discriminator.model.summary())
