#!/usr/bin/env python3
"""Convolutional back propagation."""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Perform back propagation over a convolutional layer."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
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
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            a_slice = A_prev_pad[:, vert_start:vert_end,
                                 horiz_start:horiz_end, :]

            for k in range(c_new):
                dz = dZ[:, i, j, k][:, None, None, None]
                dW[:, :, :, k] += np.sum(a_slice * dz, axis=0)
                dA_prev_pad[:, vert_start:vert_end,
                           horiz_start:horiz_end, :] += (
                    W[:, :, :, k] * dz
                )

    if ph == 0:
        h_slice = slice(None)
    else:
        h_slice = slice(ph, -ph)

    if pw == 0:
        w_slice = slice(None)
    else:
        w_slice = slice(pw, -pw)

    dA_prev = dA_prev_pad[:, h_slice, w_slice, :]

    return dA_prev, dW, db
