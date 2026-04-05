#!/usr/bin/env python3
"""Pooling back propagation."""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform back propagation over a pooling layer."""
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            a_slice = A_prev[:, vert_start:vert_end,
                             horiz_start:horiz_end, :]

            if mode == 'max':
                mask = (a_slice == np.max(a_slice,
                                          axis=(1, 2),
                                          keepdims=True))
                dA_prev[:, vert_start:vert_end,
                        horiz_start:horiz_end, :] += (
                    mask * dA[:, i:i + 1, j:j + 1, :]
                )
            else:
                dA_prev[:, vert_start:vert_end,
                        horiz_start:horiz_end, :] += (
                    dA[:, i:i + 1, j:j + 1, :] / (kh * kw)
                )

    return dA_prev
