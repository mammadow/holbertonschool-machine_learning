#!/usr/bin/env python3
"""Variational Autoencoder"""
import tensorflow.keras as keras


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
        mean, log_var = args
        batch = keras.backend.shape(mean)[0]
        epsilon = keras.backend.random_normal(shape=(batch, latent_dims))
        return mean + keras.backend.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,))([mean, log_var])
    encoder = keras.Model(inputs, [z, mean, log_var])

    # ---------- Decoder ----------
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)

    # ---------- Full autoencoder ----------
    auto_outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, auto_outputs)

    def vae_loss(y_true, y_pred):
        """Reconstruction (BCE summed over inputs) + KL divergence."""
        reconstruction = keras.backend.binary_crossentropy(y_true, y_pred)
        reconstruction = keras.backend.sum(reconstruction, axis=1)
        kl = 1 + log_var - keras.backend.square(mean)
        kl = kl - keras.backend.exp(log_var)
        kl = -0.5 * keras.backend.sum(kl, axis=1)
        return reconstruction + kl

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
