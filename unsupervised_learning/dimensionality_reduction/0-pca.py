#!/usr/bin/env python3
"""Performs PCA on a dataset using SVD."""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset using SVD."""
    u, s, vh = np.linalg.svd(X)
    cum = np.cumsum(s)
    thresh = cum[len(cum) - 1] * var
    mask = np.where(thresh > cum)
    var = cum[mask]
    idx = len(var) + 1
    W = vh.T
    Wr = W[:, 0:idx]
    return Wr
