import tensorflow as tf


class ConditionalGAN(tf.keras.Model):
    def __init__(self, image_size, num_classes, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    @property
    def metrics(self):
        return [self.generator.gen_loss_tracker, self.discriminator.disc_loss_tracker]

    def compile(self):
        super(ConditionalGAN, self).compile()

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[self.image_size[0] * self.image_size[1]]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.image_size[0], self.image_size[1], self.num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator.generate(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        self.discriminator.train(images=combined_images, labels=labels)

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        self.generator.train(random_vector=random_vector_labels,
                             image_hot_labels=image_one_hot_labels,
                             misleading_labels=misleading_labels,
                             discriminator=self.discriminator)

        # Monitor loss.
        self.generator.update_weights()
        self.discriminator.update_weights()

        return {
            "g_loss": self.generator.get_loss_track(),
            "d_loss": self.discriminator.get_loss_track(),
        }