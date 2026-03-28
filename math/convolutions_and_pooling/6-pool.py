#!/usr/bin/env python3
"""Pooling on images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Perform pooling on images."""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = ((h - kh) // sh) + 1
    ow = ((w - kw) // sw) + 1
    output = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            window = images[:, i * sh:i * sh + kh,
                            j * sw:j * sw + kw, :]

            if mode == 'max':
                output[:, i, j, :] = np.max(window, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(window, axis=(1, 2))

    return output
