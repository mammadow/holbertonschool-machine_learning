#!/usr/bin/env python3
"""PCA color module"""

import tensorflow as tf


def pca_color(image, alphas):
    """Perform PCA color augmentation"""
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)

    pixels = tf.reshape(image, (-1, 3))

    mean = tf.reduce_mean(pixels, axis=0)
    std = tf.math.reduce_std(pixels, axis=0)

    norm = (pixels - mean) / std

    cov = tf.matmul(norm, norm, transpose_a=True)
    cov /= tf.cast(tf.shape(norm)[0] - 1, tf.float32)

    eigvals, eigvecs = tf.linalg.eigh(cov)
    eigvals = tf.reverse(eigvals, axis=[0])
    eigvecs = tf.reverse(eigvecs, axis=[1])

    alphas = tf.cast(alphas, tf.float32)

    delta = tf.matmul(eigvecs, tf.reshape(alphas * eigvals, (3, 1)))
    delta = tf.reshape(delta, (1, 3))

    augmented = norm + delta
    augmented = augmented * std + mean

    augmented = tf.reshape(augmented, shape)
    augmented = tf.clip_by_value(augmented, 0, 255)

    return tf.cast(augmented, tf.uint8)
