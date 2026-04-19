#!/usr/bin/env python3
"""Transition layer for DenseNet"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer"""
    init = K.initializers.he_normal(seed=0)
    nb_filters = int(nb_filters * compression)

    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(nb_filters, (1, 1),
                        padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(X)

    return X, nb_filters
