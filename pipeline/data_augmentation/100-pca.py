#!/usr/bin/env python3
"""PCA color module"""

import numpy as np


def pca_color(image, alphas):
    """Perform PCA color augmentation"""
    image = image.numpy()
    img = image.reshape(-1, 3).astype(np.float32)

    scaling_factor = np.sqrt(3.0 / np.sum(np.var(img, axis=0)))
    img *= scaling_factor

    cov = np.cov(img, rowvar=False)
    u, s, v = np.linalg.svd(cov)

    rand = np.random.randn(3) * 0.1
    delta = np.dot(u, rand * s)
    delta = (delta * alphas * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

    img_out = np.clip(image + delta, 0, 255).astype(np.uint8)
    return img_out


if __name__ == "__main__":
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras.utils import save_img

    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(100)
    np.random.seed(100)

    doggies = tfds.load("stanford_dogs", split="train", as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        save_img("./images/before_pca.jpg", image)
        alphas = np.random.normal(0, 0.1, 3)
        save_img("./images/pca.jpg", pca_color(image, alphas))
