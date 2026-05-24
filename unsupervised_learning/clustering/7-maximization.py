#!/usr/bin/env python3
"""Maximization step"""

import numpy as np


def maximization(X, g):
    """calculates the maximization step"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    if np.any(g < 0) or np.any(g > 1):
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), np.ones(n)):
        return None, None, None

    try:
        Nk = np.sum(g, axis=1)

        pi = Nk / n

        m = (g @ X) / Nk[:, np.newaxis]

        S = np.zeros((k, d, d))

        for i in range(k):
            diff = X - m[i]
            weighted = g[i][:, np.newaxis] * diff
            S[i] = (weighted.T @ diff) / Nk[i]

        return pi, m, S

    except Exception:
        return None, None, None
