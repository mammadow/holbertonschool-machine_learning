#!/usr/bin/env python3
"""Creates a batch normalization layer for a neural network"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network"""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    x = tf.keras.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=initializer
    )(prev)

    x = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        beta_initializer='zeros',
        gamma_initializer='ones'
    )(x)

    return activation(x)
