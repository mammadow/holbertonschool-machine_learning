#!/usr/bin/env python3
"""Computes gradients for t-SNE."""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculates gradients of Y and Q affinities."""
    Q, num = Q_affinities(Y)

    diff = Y[:, None, :] - Y[None, :, :]

    PQ = P - Q
    sym = PQ + PQ.T

    dY = np.sum(sym[:, :, None] * num[:, :, None] * diff, axis=1)

    return dY, Q
