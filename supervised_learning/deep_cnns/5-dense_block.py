#!/usr/bin/env python3
"""Dense Block"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block"""
    init = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        X1 = K.layers.BatchNormalization(axis=3)(X)
        X1 = K.layers.ReLU()(X1)
        X1 = K.layers.Conv2D(4 * growth_rate, (1, 1),
                             padding='same',
                             kernel_initializer=init)(X1)

        X1 = K.layers.BatchNormalization(axis=3)(X1)
        X1 = K.layers.ReLU()(X1)
        X1 = K.layers.Conv2D(growth_rate, (3, 3),
                             padding='same',
                             kernel_initializer=init)(X1)

        X = K.layers.Concatenate(axis=3)([X, X1])
        nb_filters += growth_rate

    return X, nb_filters
