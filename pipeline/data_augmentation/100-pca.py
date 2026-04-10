#!/usr/bin/env python3
"""PCA color module"""

import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """Perform PCA color augmentation"""
    renorm_image = np.reshape(image, (image.shape[0] * image.shape[1], 3))

    mean = np.mean(renorm_image, axis=0)
    std = np.std(renorm_image, axis=0)

    renorm_image = renorm_image.astype('float32')
    renorm_image -= np.mean(renorm_image, axis=0)
    renorm_image /= np.std(renorm_image, axis=0)

    cov = np.cov(renorm_image, rowvar=False)

    lambdas, p = np.linalg.eig(cov)

    delta = np.dot(p, alphas * lambdas)

    pca_augmentation = renorm_image + delta
    pca_color_image = pca_augmentation * std + mean
    pca_color_image = pca_color_image.reshape(image.shape[0], image.shape[1], 3)
    pca_color_image = np.maximum(np.minimum(pca_color_image, 255), 0).astype(
        'uint8'
    )

    return pca_color_image
