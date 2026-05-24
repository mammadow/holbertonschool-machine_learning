#!/usr/bin/env python3
"""Performs the expectation step in the EM algorithm for a GMM"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates expectation step for a GMM"""
    if (not isinstance(X, np.ndarray) or not isinstance(pi, np.ndarray) or
        not isinstance(m, np.ndarray) or not isinstance(S, np.ndarray)):
        return None, None
    if X.ndim != 2 or pi.ndim != 1 or m.ndim != 2 or S.ndim != 3:
        return None, None
    if X.shape[0] == 0 or pi.shape[0] != m.shape[0] or S.shape[0] != m.shape[0]:
        return None, None
    if m.shape[1] != X.shape[1] or S.shape[1] != S.shape[2]:
        return None, None

    k, n = pi.shape[0], X.shape[0]

    g = np.zeros((k, n))

    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])

    total = np.sum(g, axis=0, keepdims=True)

    g = g / total

    l = np.sum(np.log(total))

    return g, l
