#!/usr/bin/env python3
"""PCA color module"""

import tensorflow as tf


def pca_color(image, alphas):
    """Perform PCA color augmentation"""
    tnp = tf.experimental.numpy

    renorm_image = tnp.reshape(
        image, (image.shape[0] * image.shape[1], 3)
    )

    mean = tnp.mean(renorm_image, axis=0)
    std = tnp.std(renorm_image, axis=0)

    renorm_image = tnp.asarray(renorm_image, dtype='float32')
    renorm_image -= tnp.mean(renorm_image, axis=0)
    renorm_image /= tnp.std(renorm_image, axis=0)

    cov = tnp.cov(renorm_image, rowvar=False)
    lambdas, p = tnp.linalg.eig(cov)
    delta = tnp.dot(p, alphas * lambdas)

    pca_augmentation = renorm_image + delta
    pca_color_image = pca_augmentation * std + mean
    pca_color_image = tnp.reshape(
        pca_color_image, (image.shape[0], image.shape[1], 3)
    )
    pca_color_image = tnp.maximum(
        tnp.minimum(pca_color_image, 255), 0
    ).astype('uint8')

    return pca_color_image
