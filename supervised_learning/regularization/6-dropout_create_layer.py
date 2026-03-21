#!/usr/bin/env python3
"""creates a layer of a neural network using dropout"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """creates a layer of a neural network using dropout"""
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode="fan_avg"
    )

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    x = dense(prev)

    if training:
        dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
        x = dropout(x, training=training)

    return x
