#!/usr/bin/env python3
"""DenseNet-121"""
from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture"""
    init = K.initializers.he_normal(seed=0)

    X = K.Input(shape=(224, 224, 3))

    # Initial layer
    X1 = K.layers.BatchNormalization(axis=3)(X)
    X1 = K.layers.Activation('relu')(X1)
    X1 = K.layers.Conv2D(64, (7, 7),
                         strides=(2, 2),
                         padding='same',
                         kernel_initializer=init)(X1)
    X1 = K.layers.MaxPooling2D((3, 3),
                               strides=(2, 2),
                               padding='same')(X1)

    nb_filters = 64

    # Dense Block 1
    X1, nb_filters = dense_block(X1, nb_filters,
                                 growth_rate, 6)
    X1, nb_filters = transition_layer(X1, nb_filters,
                                      compression)

    # Dense Block 2
    X1, nb_filters = dense_block(X1, nb_filters,
                                 growth_rate, 12)
    X1, nb_filters = transition_layer(X1, nb_filters,
                                      compression)

    # Dense Block 3
    X1, nb_filters = dense_block(X1, nb_filters,
                                 growth_rate, 24)
    X1, nb_filters = transition_layer(X1, nb_filters,
                                      compression)

    # Dense Block 4
    X1, nb_filters = dense_block(X1, nb_filters,
                                 growth_rate, 16)

    # Final layers
    X1 = K.layers.BatchNormalization(axis=3)(X1)
    X1 = K.layers.Activation('relu')(X1)
    X1 = K.layers.AveragePooling2D((7, 7))(X1)

    X1 = K.layers.Dense(1000,
                         activation='softmax',
                         kernel_initializer=init)(X1)

    model = K.models.Model(inputs=X, outputs=X1)
    return model
