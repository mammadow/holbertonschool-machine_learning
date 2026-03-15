#!/usr/bin/env python3
"""Creates a batch normalization layer for a neural network in tensorflow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    Z = tf.keras.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=init
    )(prev)

    mean, variance = tf.nn.moments(Z, axes=[0])

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    Z_norm = tf.nn.batch_normalization(
        Z,
        mean,
        variance,
        beta,
        gamma,
        1e-7
    )

    return activation(Z_norm)
