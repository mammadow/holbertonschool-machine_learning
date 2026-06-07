#!/usr/bin/env python3
"""Variational autoencoder implementation."""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder model."""

    encoder_input = keras.Input(shape=(input_dims,))

    x = encoder_input
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims)(x)
    log_var = keras.layers.Dense(latent_dims)(x)

    def sampling(args):
        """Reparameterization trick."""
        mu, log_var = args
        epsilon = K.random_normal(shape=(K.shape(mu)[0], latent_dims))
        return mu + K.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_var])

    encoder = keras.Model(encoder_input, [z, mu, log_var])

    decoder_input = keras.Input(shape=(latent_dims,))

    x = decoder_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(decoder_input, decoder_output)

    output = decoder(z)
    auto = keras.Model(encoder_input, output)

    kl_loss = -0.5 * K.sum(
        1 + log_var - K.square(mu) - K.exp(log_var),
        axis=-1
    )

    auto.add_loss(K.mean(kl_loss))

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
