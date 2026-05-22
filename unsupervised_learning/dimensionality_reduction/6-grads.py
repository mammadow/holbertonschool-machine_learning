#!/usr/bin/env python3
"""Computes gradients for t-SNE."""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculates gradients of Y and Q affinities."""
    Q, num = Q_affinities(Y)

    PQ = P - Q

    diff = Y[:, None, :] - Y[None, :, :]
    mult = (PQ + PQ.T) * num

    dY = np.sum(mult[:, :, None] * diff, axis=1)

    return dY, Q
