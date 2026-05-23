#!/usr/bin/env python3
"""Performs K-means clustering"""

import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    try:
        C = initialize(X, k)
        if C is None:
            return None, None

        for _ in range(iterations):

            # --- Assignment step ---
            dists = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
            clss = np.argmin(dists, axis=1)

            new_C = np.zeros_like(C)

            # --- Update step ---
            for i in range(k):
                points = X[clss == i]

                if len(points) == 0:
                    new_C[i] = initialize(X, 1)
                else:
                    new_C[i] = points.mean(axis=0)

            # --- Convergence check ---
            if np.allclose(C, new_C):
                C = new_C
                break

            C = new_C

        # final assignment
        dists = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(dists, axis=1)

        return C, clss

    except Exception:
        return None, None
