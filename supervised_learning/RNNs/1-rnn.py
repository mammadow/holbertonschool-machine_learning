#!/usr/bin/env python3
"""RNN forward propagation."""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN."""
    t, m, h = X.shape[0], X.shape[1], h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    H[0] = h_0

    for i in range(t):
        h_next, y = rnn_cell.forward(H[i], X[i])
        H[i + 1] = h_next
        Y[i] = y

    return H, Y
