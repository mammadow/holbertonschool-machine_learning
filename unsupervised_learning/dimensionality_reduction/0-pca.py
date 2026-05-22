#!/usr/bin/env python3
import numpy as np


def pca(X, var=0.95):
    """Performs PCA and returns projection matrix."""
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = S ** 2
    vr = ev / np.sum(ev)
    cvr = np.cumsum(vr)

    nd = 0
    for v in cvr:
        nd += 1
        if v >= var:
            break

    return Vt.T[:, :nd]
