#!/usr/bin/env python3
"""Calculates the gradients"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculates gradients of Y and Q affinities."""
    n, ndim = Y.shape
    dY = np.zeros((n, ndim))

    Q, num = Q_affinities(Y)
    PQ = P - Q

    for i in range(n):
        weights = PQ[:, i] * num[:, i]
        dY[i] = np.sum(np.tile(weights, (ndim, 1)).T * (Y[i] - Y), axis=0)

    return dY, Q
