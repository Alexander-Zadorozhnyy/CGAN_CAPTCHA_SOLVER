import tensorflow as tf
from tensorflow.keras import layers


class Generator:
    def __init__(self, in_channels, optimizer, loss_fn, height, width):
        self.in_channels = in_channels
        self.height = height
        self.width = width

        self.g_optimizer = optimizer
        self.loss_fn = loss_fn
        self.g_loss = None
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")

        self.model = tf.keras.Sequential(
            [
                layers.InputLayer((self.in_channels,)),
                # We want to generate 128 + num_classes coefficients to reshape into a
                # 7x7x(128 + num_classes) map.
                layers.Dense((self.height // 4) * (self.width // 4) * self.in_channels),
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape(((self.height // 4), (self.width // 4), self.in_channels)),

                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),

                # layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding="same"),
                # layers.BatchNormalization(),
                # layers.LeakyReLU(alpha=0.2),

                layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            ],
            name="generator",
        )

    def generate(self, x):
        return self.model(x)

    def train(self, random_vector, image_hot_labels, misleading_labels, discriminator):
        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generate(random_vector)
            fake_image_and_labels = tf.concat([fake_images, image_hot_labels], -1)
            predictions = discriminator.predict(fake_image_and_labels)
            self.g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(self.g_loss, self.model.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def update_weights(self):
        self.gen_loss_tracker.update_state(self.g_loss)

    def get_loss_track(self):
        return self.gen_loss_tracker.result()


if __name__ == "__main__":
    generator = Generator(in_channels=100,
                          optimizer=None,
                          loss_fn=tf.keras.losses.BinaryCrossentropy(),
                          height=40,
                          width=24)
    print(generator.model.summary())
