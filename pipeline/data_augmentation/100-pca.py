#!/usr/bin/env python3
"""PCA color module"""

import tensorflow as tf


def pca_color(image, alphas):
    """Perform PCA color augmentation"""
    image = tf.cast(image, tf.float32)
    flat = tf.reshape(image, (-1, 3))

    var = tf.math.reduce_variance(flat, axis=0)
    scaling_factor = tf.sqrt(3.0 / tf.reduce_sum(var))
    flat = flat * scaling_factor

    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean

    n = tf.cast(tf.shape(centered)[0], tf.float32)
    cov = tf.matmul(centered, centered, transpose_a=True) / (n - 1.0)

    s, u, _ = tf.linalg.svd(cov)

    alphas = tf.cast(alphas, tf.float32)
    delta = tf.linalg.matvec(u, alphas * s)
    delta = tf.cast(delta * 255.0, tf.int32)
    delta = tf.reshape(delta, (1, 1, 3))

    out = tf.cast(image, tf.int32) + delta
    out = tf.clip_by_value(out, 0, 255)

    return tf.cast(out, tf.uint8)
