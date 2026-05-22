#!/usr/bin/env python3
"""Performs PCA on a dataset using SVD."""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset using SVD."""
    u, s, vh = np.linalg.svd(X)
    ev = s ** 2
    cum = np.cumsum(ev)
    thresh = cum[-1] * var
    idx = np.where(cum >= thresh)[0][0] + 1
    return vh.T[:, :idx]
