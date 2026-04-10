#!/usr/bin/env python3
"""PCA color module"""

import tensorflow as tf


def pca_color(image, alphas):
    """Perform PCA color augmentation"""
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)

    pixels = tf.reshape(image, (-1, 3))
    mean = tf.reduce_mean(pixels, axis=0)
    centered = pixels - mean

    cov = tf.matmul(centered, centered, transpose_a=True)
    cov /= tf.cast(tf.shape(centered)[0] - 1, tf.float32)

    eigvals, eigvecs = tf.linalg.eigh(cov)
    eigvals = tf.reshape(eigvals, (1, 3))
    alphas = tf.cast(tf.reshape(alphas, (1, 3)), tf.float32)

    delta = tf.matmul(eigvecs, tf.transpose(alphas * eigvals))
    delta = tf.reshape(delta, (1, 3))

    augmented = pixels + delta
    augmented = tf.reshape(augmented, shape)
    augmented = tf.clip_by_value(augmented, 0, 255)

    return tf.cast(augmented, tf.uint8)
