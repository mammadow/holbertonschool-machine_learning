#!/usr/bin/env python3
"""Dense Block"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block"""
    init = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        # Bottleneck layer: BN → ReLU → 1x1 Conv (4 * growth_rate)
        X_bn = K.layers.BatchNormalization(axis=3)(X)
        X_act = K.layers.Activation('relu')(X_bn)
        X_conv = K.layers.Conv2D(4 * growth_rate,
                                 (1, 1),
                                 padding='same',
                                 kernel_initializer=init)(X_act)

        # Second layer: BN → ReLU → 3x3 Conv (growth_rate)
        X_bn2 = K.layers.BatchNormalization(axis=3)(X_conv)
        X_act2 = K.layers.Activation('relu')(X_bn2)
        X_conv2 = K.layers.Conv2D(growth_rate,
                                  (3, 3),
                                  padding='same',
                                  kernel_initializer=init)(X_act2)

        # Concatenate with input
        X = K.layers.Concatenate(axis=3)([X, X_conv2])

        # Update filter count
        nb_filters += growth_rate

    return X, nb_filters
