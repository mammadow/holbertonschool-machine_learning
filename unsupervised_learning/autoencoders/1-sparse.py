#!/usr/bin/env python3
"""Sparse autoencoder implementation."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder model."""

    input_layer = keras.Input(shape=(input_dims,))

    x = input_layer
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)

    encoder = keras.Model(inputs=input_layer, outputs=latent)

    latent_input = keras.Input(shape=(latent_dims,))
    x = latent_input

    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    output = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(inputs=latent_input, outputs=output)

    reconstructed = decoder(encoder(input_layer))
    auto = keras.Model(inputs=input_layer, outputs=reconstructed)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
