#!/usr/bin/env python3
"""Performs the maximization step in the EM algorithm for a GMM"""

import numpy as np


def maximization(X, g):
    """Calculates the maximization step for a GMM"""
    if not isinstance(X, np.ndarray) or not isinstance(g, np.ndarray):
        return None, None, None

    if X.ndim != 2 or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), np.ones(n)):
        return None, None, None

    if np.any(g < 0) or np.any(g > 1):
        return None, None, None

    nk = np.sum(g, axis=1)

    pi = nk / n

    m = (g @ X) / nk[:, np.newaxis]

    S = np.zeros((k, d, d))

    for i in range(k):
        diff = X - m[i]
        weighted = g[i][:, np.newaxis] * diff
        S[i] = (weighted.T @ diff) / nk[i]

    return pi, m, S
