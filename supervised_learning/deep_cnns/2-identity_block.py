#!/usr/bin/env python3
"""Identity block for ResNet"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block as described in Resnet"""
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=0)
    X = K.layers.Conv2D(filters=F11,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.ReLU()(X)

    X = K.layers.Conv2D(filters=F3,
                        kernel_size=(3, 3),
                        padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.ReLU()(X)

    X = K.layers.Conv2D(filters=F12,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Add()([X, A_prev])
    X = K.layers.ReLU()(X)

    return X
