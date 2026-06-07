#!/usr/bin/env python3
"""Convolutional autoencoder implementation."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder model."""

    encoder_input = keras.Input(shape=input_dims)

    x = encoder_input
    for filt in filters:
        x = keras.layers.Conv2D(
            filt,
            (3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(
            (2, 2),
            padding='same'
        )(x)

    encoder = keras.Model(inputs=encoder_input, outputs=x)

    decoder_input = keras.Input(shape=latent_dims)

    x = decoder_input
    rev_filters = filters[::-1]

    for filt in rev_filters[:-1]:
        x = keras.layers.Conv2D(
            filt,
            (3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        rev_filters[-1],
        (3, 3),
        padding='valid',
        activation='relu'
    )(x)

    x = keras.layers.UpSampling2D((2, 2))(x)

    output = keras.layers.Conv2D(
        input_dims[-1],
        (3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(inputs=decoder_input, outputs=output)

    auto_input = encoder_input
    reconstructed = decoder(encoder(auto_input))

    auto = keras.Model(inputs=auto_input, outputs=reconstructed)

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
