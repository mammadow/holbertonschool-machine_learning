#!/usr/bin/env python3
"""Initializes GMM parameters"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    try:
        n, d = X.shape

        pi = np.full(k, 1 / k)

        C, _ = kmeans(X, k)

        m = C

        S = np.ones((k, d, d))
        for i in range(k):
            S[i] = np.eye(d)

        return pi, m, S

    except Exception:
        return None, None, None
