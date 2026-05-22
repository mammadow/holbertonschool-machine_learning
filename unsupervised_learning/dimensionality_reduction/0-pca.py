#!/usr/bin/env python3
"""PCA implementation using SVD"""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset."""
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = S ** 2
    vr = ev / np.sum(ev)
    cvr = np.cumsum(vr)
    nd = np.argmax(cvr >= var) + 1
    return Vt.T[:, :nd]
