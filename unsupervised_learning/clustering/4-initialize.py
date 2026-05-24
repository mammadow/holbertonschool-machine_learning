#!/usr/bin/env python3
"""Initializes variables for a Gaussian Mixture Model"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes GMM parameters using K-means"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    try:
        n, d = X.shape

        pi = np.full(k, 1 / k)

        m, _ = kmeans(X, k)

        S = np.tile(np.eye(d), (k, 1, 1))

        return pi, m, S

    except Exception:
        return None, None, None
