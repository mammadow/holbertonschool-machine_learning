#!/usr/bin/env python3
"""Convolutional forward propagation."""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same",
                 stride=(1, 1)):
    """Perform forward propagation over a convolutional layer."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = (((h_prev - 1) * sh + kh - h_prev) + 1) // 2
        pw = (((w_prev - 1) * sw + kw - w_prev) + 1) // 2
    else:
        ph = 0
        pw = 0

    A_prev_pad = np.pad(A_prev,
                        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')

    h_new = ((h_prev + 2 * ph - kh) // sh) + 1
    w_new = ((w_prev + 2 * pw - kw) // sw) + 1
    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw
            window = A_prev_pad[:, vert_start:vert_end,
                                horiz_start:horiz_end, :]

            for k in range(c_new):
                Z[:, i, j, k] = np.sum(window * W[:, :, :, k],
                                       axis=(1, 2, 3))
                Z[:, i, j, k] += b[0, 0, 0, k]

    return activation(Z)
