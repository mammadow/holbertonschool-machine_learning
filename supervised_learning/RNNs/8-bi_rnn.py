#!/usr/bin/env python3
"""Bidirectional RNN module"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward propagation for a bidirectional RNN"""
    t, m, i = X.shape
    h = h_0.shape[1]

    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))

    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        Hf[step] = h_prev

    h_next = h_t
    for step in range(t - 1, -1, -1):
        h_next = bi_cell.backward(h_next, X[step])
        Hb[step] = h_next

    H = np.concatenate((Hf, Hb), axis=2)
    Y = bi_cell.output(H)

    return H, Y
