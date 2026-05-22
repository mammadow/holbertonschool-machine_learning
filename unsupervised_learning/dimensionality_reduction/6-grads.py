#!/usr/bin/env python3
"""Computes gradients for t-SNE."""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculates gradients of Y and Q affinities."""
    Q, num = Q_affinities(Y)

    PQ = P - Q
    n, ndim = Y.shape

    dY = np.zeros((n, ndim))

    for i in range(n):
        diff = Y[i] - Y
        mult = (PQ[i] + PQ[:, i])[:, None] * num[i][:, None]
        dY[i] = np.sum(mult * diff, axis=0)

    return dY, Q
