#!/usr/bin/env python3
"""Initialize K-means centroids"""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means"""
    if not isinstance(X, np.ndarray) or X.ndim != 2 or k <= 0:
        return None

    try:
        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        return np.random.uniform(mins, maxs, size=(k, X.shape[1]))
    except Exception:
        return None
