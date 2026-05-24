#!/usr/bin/env python3
"""Performs the expectation step in the EM algorithm for a GMM"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step for a GMM"""
    if (not isinstance(X, np.ndarray) or
            not isinstance(pi, np.ndarray) or
            not isinstance(m, np.ndarray) or
            not isinstance(S, np.ndarray)):
        return None, None

    if X.ndim != 2 or pi.ndim != 1 or m.ndim != 2 or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d):
        return None, None

    if S.shape != (k, d, d):
        return None, None

    if not np.isclose(np.sum(pi), 1):
        return None, None

    if np.any(pi < 0):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])

        if P is None:
            return None, None

        g[i] = pi[i] * P

    total = np.sum(g, axis=0)

    if np.any(total == 0):
        return None, None

    log_likelihood = np.sum(np.log(total))

    g = g / total

    return g, log_likelihood
