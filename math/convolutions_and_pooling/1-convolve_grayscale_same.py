#!/usr/bin/env python3
"""Same convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Perform same convolution."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = kh // 2
    pw = kw // 2

    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            window = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(window * kernel,
                                     axis=(1, 2))

    return output
