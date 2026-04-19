#!/usr/bin/env python3
"""Inception block"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Builds an inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters

    path1 = K.layers.Conv2D(F1, (1, 1),
                            padding='same',
                            activation='relu')(A_prev)

    path2 = K.layers.Conv2D(F3R, (1, 1),
                            padding='same',
                            activation='relu')(A_prev)
    path2 = K.layers.Conv2D(F3, (3, 3),
                            padding='same',
                            activation='relu')(path2)

    path3 = K.layers.Conv2D(F5R, (1, 1),
                            padding='same',
                            activation='relu')(A_prev)
    path3 = K.layers.Conv2D(F5, (5, 5),
                            padding='same',
                            activation='relu')(path3)

    path4 = K.layers.MaxPooling2D((3, 3),
                                  strides=(1, 1),
                                  padding='same')(A_prev)
    path4 = K.layers.Conv2D(FPP, (1, 1),
                            padding='same',
                            activation='relu')(path4)

    output = K.layers.Concatenate(axis=3)([path1, path2, path3, path4])

    return output
