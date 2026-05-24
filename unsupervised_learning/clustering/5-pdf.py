#!/usr/bin/env python3
"""Calculates the probability density function of a Gaussian distribution"""

import numpy as np


def pdf(X, m, S):
    """Computes the PDF of a multivariate Gaussian distribution"""
    if not isinstance(X, np.ndarray) or not isinstance(m, np.ndarray):
        return None
    if not isinstance(S, np.ndarray):
        return None
    if X.ndim != 2 or m.ndim != 1 or S.ndim != 2:
        return None
    if X.shape[1] != m.shape[0] or S.shape[0] != S.shape[1]:
        return None
    if S.shape[0] != m.shape[0]:
        return None

    d = m.shape[0]

    det = np.linalg.det(S)
    if det <= 0:
        return None

    inv = np.linalg.inv(S)

    X_shifted = X - m

    exp_term = np.sum((X_shifted @ inv) * X_shifted, axis=1)

    denom = np.sqrt(((2 * np.pi) ** d) * det)

    P = (1 / denom) * np.exp(-0.5 * exp_term)

    return np.maximum(P, 1e-300)
