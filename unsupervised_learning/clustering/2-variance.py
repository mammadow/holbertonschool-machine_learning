#!/usr/bin/env python3
"""Calculates total intra-cluster variance"""

import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance"""
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if X.ndim != 2 or C.ndim != 2:
        return None

    try:
        diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
        dist = np.sum(diff ** 2, axis=2)
        min_dist = np.min(dist, axis=1)

        return np.sum(min_dist)

    except Exception:
        return None
