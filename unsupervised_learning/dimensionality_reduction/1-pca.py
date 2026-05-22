#!/usr/bin/env python3
"""Performs PCA to reduce dimensionality using SVD."""
import numpy as np


def pca(X, ndim):
    """Performs PCA and returns transformed dataset."""
    X = X - np.mean(X, axis=0)
    _, s, vh = np.linalg.svd(X, full_matrices=False)
    W = vh.T[:, :ndim]
    T = np.matmul(X, W)
    return T
