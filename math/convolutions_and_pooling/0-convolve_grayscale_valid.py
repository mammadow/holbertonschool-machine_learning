#!/usr/bin/env python3
"""Performs a valid convolution on multiple grayscale images using a given kernel."""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Applies a valid convolution to a batch of grayscale images and returns the resulting feature maps."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            window = images[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
