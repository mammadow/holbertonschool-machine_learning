#!/usr/bin/env python3
"""Builds a neural network with the Keras library"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network using Sequential"""
    model = K.models.Sequential()

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
            ))

        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
