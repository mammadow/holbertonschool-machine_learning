#!/usr/bin/env python3
"""PCA color module"""

import tensorflow as tf


def pca_color(image, alphas):
    """Perform PCA color augmentation"""
    image = tf.cast(image, tf.float32)

    flat = tf.reshape(image, (-1, 3))

    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean

    n = tf.cast(tf.shape(centered)[0], tf.float32)
    cov = tf.matmul(centered, centered, transpose_a=True) / (n - 1)

    eigvals, eigvecs = tf.linalg.eigh(cov)

    eigvals = tf.reverse(eigvals, axis=[0])
    eigvecs = tf.reverse(eigvecs, axis=[1])
    alphas = tf.cast(alphas, tf.float32)
    delta = tf.matmul(eigvecs, tf.reshape(alphas * eigvals, (3, 1)))
    delta = tf.reshape(delta, (1, 3))

    flat = flat + delta
    out = tf.reshape(flat, tf.shape(image))

    out = tf.clip_by_value(out, 0, 255)

    return tf.cast(out, tf.uint8)
