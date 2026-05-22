#!/usr/bin/env python3
"""Computes Q affinities for t-SNE."""
import numpy as np


def Q_affinities(Y):
    """Calculates Q affinities and numerator matrix."""
    sum_Y = np.sum(np.square(Y), axis=1)
    D = sum_Y[:, None] + sum_Y[None, :] - 2 * np.dot(Y, Y.T)

    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)

    Q = num / np.sum(num)

    return Q, num
