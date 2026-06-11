#!/usr/bin/env python3
"""Variational Autoencoder"""
import tensorflow.keras as keras
K = keras.backend


class KLDivergence(keras.layers.Layer):
    """Layer that adds the KL divergence to the model's losses."""

    def call(self, inputs):
        """Compute KL from [mean, log_var] and register it as a loss."""
        mean, log_var = inputs
        kl = 1 + log_var - K.square(mean) - K.exp(log_var)
        kl = -0.5 * K.sum(kl, axis=1)
        self.add_loss(K.mean(kl))
        return mean


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder.
    """
    # ---------- Encoder ----------
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Reparameterization trick."""
        mu, lv = args
        batch = K.shape(mu)[0]
        epsilon = K.random_normal(shape=(batch, latent_dims))
        return mu + K.exp(lv / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mean, log_var])
    encoder = keras.Model(inputs, [z, mean, log_var])

    # ---------- Decoder ----------
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)

    # ---------- Full autoencoder ----------
    enc_z, enc_mean, enc_log_var = encoder(inputs)
    KLDivergence()([enc_mean, enc_log_var])  # registers KL loss
    auto_outputs = decoder(enc_z)
    auto = keras.Model(inputs, auto_outputs)

    def vae_loss(y_true, y_pred):
        """Reconstruction BCE, summed over inputs."""
        bce = K.binary_crossentropy(y_true, y_pred)
        return K.sum(bce, axis=1)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
