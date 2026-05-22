#!/usr/bin/env python3
"""Initializes variables for t-SNE P matrix computation."""
import numpy as np


def P_init(X, perplexity):
    """Initializes D, P, betas, and H for t-SNE."""
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X[:, None] + sum_X[None, :] - 2 * np.dot(X, X.T)
    np.fill_diagonal(D, 0)

    n = X.shape[0]
    P = np.zeros((n, n))
    betas = np.ones((n, 1))

    H = np.log2(perplexity)

    return D, P, betas, H
