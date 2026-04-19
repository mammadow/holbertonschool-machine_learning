#!/usr/bin/env python3
"""Inception Network"""
from tensorflow import keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the Inception network"""
    X = K.Input(shape=(224, 224, 3))

    # Initial layers
    X1 = K.layers.Conv2D(64, (7, 7),
                         strides=(2, 2),
                         padding='same',
                         activation='relu')(X)
    X1 = K.layers.MaxPooling2D((3, 3),
                               strides=(2, 2),
                               padding='same')(X1)

    X1 = K.layers.Conv2D(64, (1, 1),
                         padding='same',
                         activation='relu')(X1)
    X1 = K.layers.Conv2D(192, (3, 3),
                         padding='same',
                         activation='relu')(X1)
    X1 = K.layers.MaxPooling2D((3, 3),
                               strides=(2, 2),
                               padding='same')(X1)

    # Inception blocks
    X1 = inception_block(X1, [64, 96, 128, 16, 32, 32])
    X1 = inception_block(X1, [128, 128, 192, 32, 96, 64])
    X1 = K.layers.MaxPooling2D((3, 3),
                               strides=(2, 2),
                               padding='same')(X1)

    X1 = inception_block(X1, [192, 96, 208, 16, 48, 64])
    X1 = inception_block(X1, [160, 112, 224, 24, 64, 64])
    X1 = inception_block(X1, [128, 128, 256, 24, 64, 64])
    X1 = inception_block(X1, [112, 144, 288, 32, 64, 64])
    X1 = inception_block(X1, [256, 160, 320, 32, 128, 128])
    X1 = K.layers.MaxPooling2D((3, 3),
                               strides=(2, 2),
                               padding='same')(X1)

    X1 = inception_block(X1, [256, 160, 320, 32, 128, 128])
    X1 = inception_block(X1, [384, 192, 384, 48, 128, 128])

    # Final layers
    X1 = K.layers.AveragePooling2D((7, 7))(X1)
    X1 = K.layers.Dropout(0.4)(X1)
    X1 = K.layers.Dense(1000,
                        activation='softmax')(X1)

    model = K.models.Model(inputs=X, outputs=X1)
    return model
