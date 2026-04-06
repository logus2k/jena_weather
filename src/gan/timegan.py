"""
TimeGAN implementation for synthetic time series generation.

Based on Yoon et al. (2019) "Time-series Generative Adversarial Networks".
The model consists of five GRU-based sub-networks (embedder, recovery,
generator, supervisor, discriminator) trained in three phases:
  1. Autoencoder pretraining (embedder + recovery)
  2. Supervisor pretraining (temporal dynamics in latent space)
  3. Adversarial training (generator vs discriminator)

Reference: https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d3eed8-Abstract.html
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TimeGAN:
    def __init__(
        self,
        seq_len,
        n_features,
        hidden_dim=24,
        num_layers=3,
        learning_rate=1e-4,
        gamma=1.0,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.embedder = self._build_rnn_network(
            output_dim=hidden_dim,
            input_dim=n_features,
            name="embedder",
        )
        self.recovery = self._build_rnn_network(
            output_dim=n_features,
            input_dim=hidden_dim,
            name="recovery",
        )
        self.generator = self._build_rnn_network(
            output_dim=hidden_dim,
            input_dim=hidden_dim,
            name="generator",
        )
        self.supervisor = self._build_rnn_network(
            output_dim=hidden_dim,
            input_dim=hidden_dim,
            name="supervisor",
        )
        self.discriminator = self._build_rnn_network(
            output_dim=1,
            input_dim=hidden_dim,
            name="discriminator",
        )

    def _build_rnn_network(self, output_dim, input_dim, name="rnn_block"):
        inputs = keras.Input(shape=(self.seq_len, input_dim))
        x = inputs

        for i in range(self.num_layers):
            x = layers.GRU(
                self.hidden_dim,
                return_sequences=True,
                name=f"{name}_gru_{i+1}",
            )(x)

        outputs = layers.TimeDistributed(
            layers.Dense(output_dim),
            name=f"{name}_out",
        )(x)

        return keras.Model(inputs, outputs, name=name)

    def _build_autoencoder(self):
        X = keras.Input(shape=(self.seq_len, self.n_features))
        H = self.embedder(X)
        X_tilde = self.recovery(H)

        model = keras.Model(X, X_tilde, name="autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="mse",
        )
        return model

    def pretrain_autoencoder(self, sequences, epochs=50, batch_size=128, verbose=1):
        self.autoencoder = self._build_autoencoder()

        history = self.autoencoder.fit(
            sequences,
            sequences,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True,
        )
        return history

    def _build_supervisor_trainer(self):
        H = keras.Input(shape=(self.seq_len - 1, self.hidden_dim))
        x = H

        for i in range(self.num_layers):
            x = layers.GRU(
                self.hidden_dim,
                return_sequences=True,
                name=f"supervisor_train_gru_{i+1}",
            )(x)

        H_hat = layers.TimeDistributed(
            layers.Dense(self.hidden_dim),
            name="supervisor_train_out",
        )(x)

        model = keras.Model(H, H_hat, name="supervisor_trainer")
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="mse",
        )
        return model

    def pretrain_supervisor(self, sequences, epochs=50, batch_size=128, verbose=1):
        if not hasattr(self, "autoencoder"):
            self.autoencoder = self._build_autoencoder()

        H = self.embedder.predict(sequences, batch_size=batch_size, verbose=0)

        self.supervisor_trainer = self._build_supervisor_trainer()
        history = self.supervisor_trainer.fit(
            H[:, :-1, :],
            H[:, 1:, :],
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True,
        )
        return history

    def _build_discriminator_trainer(self):
        H_real = keras.Input(shape=(self.seq_len, self.hidden_dim))
        H_fake = keras.Input(shape=(self.seq_len, self.hidden_dim))

        Y_real = self.discriminator(H_real)
        Y_fake = self.discriminator(H_fake)

        return keras.Model(
            [H_real, H_fake],
            [Y_real, Y_fake],
            name="discriminator_trainer",
        )

    def discriminator_loss(self, y_real, y_fake):
        bce = keras.losses.BinaryCrossentropy(from_logits=True)

        real_labels = tf.ones_like(y_real)
        fake_labels = tf.zeros_like(y_fake)

        d_loss_real = bce(real_labels, y_real)
        d_loss_fake = bce(fake_labels, y_fake)

        return d_loss_real + d_loss_fake

    def _build_generator_trainer(self):
        Z = keras.Input(shape=(self.seq_len, self.hidden_dim))

        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        Y_fake = self.discriminator(H_hat)
        X_hat = self.recovery(H_hat)

        return keras.Model(Z, [Y_fake, X_hat], name="generator_trainer")

    def generator_loss(self, y_fake):
        bce = keras.losses.BinaryCrossentropy(from_logits=True)
        real_labels = tf.ones_like(y_fake)
        return bce(real_labels, y_fake)

    def prepare_adversarial_models(self):
        self.discriminator_trainer = self._build_discriminator_trainer()
        self.generator_trainer = self._build_generator_trainer()

        self.d_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=1.0,
        )
        self.g_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=1.0,
        )

    @tf.function
    def train_adversarial_step(self, X_real):
        batch_size = tf.shape(X_real)[0]

        # Real latent sequence
        H_real = self.embedder(X_real, training=False)

        # Random latent noise
        Z = tf.random.uniform(
            shape=(batch_size, self.seq_len, self.hidden_dim),
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32,
        )

        # ---------------------------
        # 1) Train discriminator
        # ---------------------------
        with tf.GradientTape() as d_tape:
            E_hat = self.generator(Z, training=True)
            H_fake = self.supervisor(E_hat, training=True)

            Y_real = self.discriminator(H_real, training=True)
            Y_fake = self.discriminator(H_fake, training=True)

            d_loss = self.discriminator_loss(Y_real, Y_fake)

        d_vars = self.discriminator.trainable_variables
        d_grads = d_tape.gradient(d_loss, d_vars)

        # train D only if it is not already too strong
        if d_loss > 0.15:
            self.d_optimizer.apply_gradients(zip(d_grads, d_vars))

        # ---------------------------
        # 2) Train generator + supervisor
        # ---------------------------
        with tf.GradientTape() as g_tape:
            # synthetic latent path
            E_hat = self.generator(Z, training=True)
            H_fake = self.supervisor(E_hat, training=True)
            X_fake = self.recovery(H_fake, training=False)

            Y_fake = self.discriminator(H_fake, training=False)

            # adversarial loss
            g_loss_u = self.generator_loss(Y_fake)

            # reconstruction-like loss in data space
            g_loss_v = tf.reduce_mean(tf.square(X_real - X_fake))

            # supervised loss in latent space
            H_hat_supervise = self.supervisor(H_real, training=True)
            g_loss_s = tf.reduce_mean(
                tf.square(H_real[:, 1:, :] - H_hat_supervise[:, :-1, :])
            )

            # total generator loss
            g_loss = g_loss_u + (100.0 * g_loss_s) + (self.gamma * g_loss_v)

        g_vars = self.generator.trainable_variables + self.supervisor.trainable_variables
        g_grads = g_tape.gradient(g_loss, g_vars)
        self.g_optimizer.apply_gradients(zip(g_grads, g_vars))

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "g_loss_u": g_loss_u,
            "g_loss_s": g_loss_s,
            "g_loss_v": g_loss_v,
        }

    def summary(self):
        print("\n=== Embedder ===")
        self.embedder.summary()
        print("\n=== Recovery ===")
        self.recovery.summary()
        print("\n=== Generator ===")
        self.generator.summary()
        print("\n=== Supervisor ===")
        self.supervisor.summary()
        print("\n=== Discriminator ===")
        self.discriminator.summary()

    def fit(self, sequences, epochs=100, batch_size=128, verbose=1):
        if not hasattr(self, "autoencoder"):
            self.autoencoder = self._build_autoencoder()

        if not hasattr(self, "supervisor_trainer"):
            self.supervisor_trainer = self._build_supervisor_trainer()

        self.prepare_adversarial_models()

        dataset = (
            tf.data.Dataset.from_tensor_slices(sequences.astype("float32"))
            .shuffle(min(len(sequences), 10000))
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        history = {
            "d_loss": [],
            "g_loss": [],
            "g_loss_u": [],
            "g_loss_s": [],
            "g_loss_v": [],
        }

        for epoch in range(epochs):
            epoch_d, epoch_g, epoch_gu, epoch_gs, epoch_gv = [], [], [], [], []

            for X_batch in dataset:
                losses = self.train_adversarial_step(X_batch)

                epoch_d.append(float(losses["d_loss"].numpy()))
                epoch_g.append(float(losses["g_loss"].numpy()))
                epoch_gu.append(float(losses["g_loss_u"].numpy()))
                epoch_gs.append(float(losses["g_loss_s"].numpy()))
                epoch_gv.append(float(losses["g_loss_v"].numpy()))

            history["d_loss"].append(np.mean(epoch_d))
            history["g_loss"].append(np.mean(epoch_g))
            history["g_loss_u"].append(np.mean(epoch_gu))
            history["g_loss_s"].append(np.mean(epoch_gs))
            history["g_loss_v"].append(np.mean(epoch_gv))

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"d_loss: {history['d_loss'][-1]:.4f} - "
                    f"g_loss: {history['g_loss'][-1]:.4f} - "
                    f"g_loss_u: {history['g_loss_u'][-1]:.4f} - "
                    f"g_loss_s: {history['g_loss_s'][-1]:.4f} - "
                    f"g_loss_v: {history['g_loss_v'][-1]:.4f}"
                )

        return history

    def generate(self, n_samples):
        Z = np.random.uniform(
            0.0,
            1.0,
            size=(n_samples, self.seq_len, self.hidden_dim),
        ).astype(np.float32)

        E_hat = self.generator.predict(Z, verbose=0)
        H_hat = self.supervisor.predict(E_hat, verbose=0)
        X_hat = self.recovery.predict(H_hat, verbose=0)

        return X_hat