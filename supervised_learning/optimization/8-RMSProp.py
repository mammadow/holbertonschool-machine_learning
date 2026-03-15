#!/usr/bin/env python3
"""Sets up the RMSProp optimization algorithm in TensorFlow"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Creates an RMSProp optimizer"""
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
