#!/usr/bin/env python3
"""PCA color module"""

import tensorflow as tf


def pca_color(image, alphas):
    """Perform PCA color augmentation"""
    tnp = tf.experimental.numpy

    image = tnp.asarray(image, dtype=tnp.float32)
    orig = tnp.array(image, copy=True)

    img = image / 255.0
    img = tnp.reshape(img, (-1, 3))
    img_centered = img - tnp.mean(img, axis=0)

    cov = tnp.cov(img_centered, rowvar=False)
    eig_vals, eig_vecs = tnp.linalg.eigh(cov)

    sort_perm = tnp.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sort_perm]
    eig_vecs = eig_vecs[:, sort_perm]

    alphas = tnp.asarray(alphas, dtype=tnp.float32)
    delta = tnp.dot(eig_vecs, alphas * eig_vals)

    for i in range(3):
        orig[..., i] += delta[i]

    orig = tnp.clip(orig, 0.0, 255.0)
    return orig.astype(tnp.uint8)
