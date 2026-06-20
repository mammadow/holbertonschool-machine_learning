#!/usr/bin/env python3
"""Deep RNN module"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN"""
    l, m, h = h_0.shape
    t = X.shape[0]

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    Y = []

    for step in range(t):
        x_t = X[step]
        for layer in range(l):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x_t)
            H[step + 1, layer] = h_next
            x_t = h_next
        Y.append(y)

    Y = np.array(Y)

    return H, Y
