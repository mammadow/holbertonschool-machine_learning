#!/usr/bin/env python3
"""Finds the optimum number of clusters by variance"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    try:
        results = []
        variances = []

        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            if C is None or clss is None:
                return None, None

            results.append((C, clss))
            variances.append(variance(X, C))

        d_vars = [variances[0] - v for v in variances]

        return results, d_vars

    except Exception:
        return None, None
