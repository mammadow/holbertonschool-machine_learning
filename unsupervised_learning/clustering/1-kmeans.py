#!/usr/bin/env python3
"""Performs K-means clustering"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    try:
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        centroids = np.random.uniform(mins, maxs, size=(k, X.shape[1]))

        for _ in range(iterations):
            old = centroids.copy()

            diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            dist = np.sum(diff ** 2, axis=2)
            clss = np.argmin(dist, axis=1)

            for i in range(k):
                pts = X[clss == i]
                if pts.shape[0] == 0:
                    centroids[i] = np.random.uniform(mins, maxs, size=X.shape[1])
                else:
                    centroids[i] = pts.mean(axis=0)

            if np.allclose(old, centroids):
                break

        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dist = np.sum(diff ** 2, axis=2)
        clss = np.argmin(dist, axis=1)

        return centroids, clss

    except Exception:
        return None, None
