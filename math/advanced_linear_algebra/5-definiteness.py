#!/usr/bin/env python3
"""5-definiteness.py"""

import numpy as np


def definiteness(matrix):
    """Determine the definiteness category of a matrix."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2:
        return None

    n, m = matrix.shape
    if n == 0 or m == 0 or n != m:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigvals = np.linalg.eigvalsh(matrix)

    tol = 1e-8
    pos = eigvals > tol
    neg = eigvals < -tol
    zero = ~(pos | neg)

    if np.all(pos):
        return "Positive definite"
    if np.all(pos | zero) and np.any(zero):
        return "Positive semi-definite"
    if np.all(neg):
        return "Negative definite"
    if np.all(neg | zero) and np.any(zero):
        return "Negative semi-definite"
    if np.any(pos) and np.any(neg):
        return "Indefinite"

    return None
