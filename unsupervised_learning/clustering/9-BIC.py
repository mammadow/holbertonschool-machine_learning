#!/usr/bin/env python3
"""
9-BIC.py
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds best number of clusters for GMM using BIC"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if not isinstance(kmin, int) or kmin <= 0 or kmin > kmax:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or kmax > n:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    ks = np.arange(kmin, kmax + 1)

    l = np.zeros(len(ks))
    b = np.zeros(len(ks))

    best_k = None
    best_bic = np.inf
    best_result = None

    all_pis, all_ms, all_Ss = [], [], []

    for i, k in enumerate(ks):
        pi, m, S, g, lkhd = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if lkhd is None:
            return None, None, None, None

        all_pis.append(pi)
        all_ms.append(m)
        all_Ss.append(S)

        l[i] = lkhd

        p = (k - 1) + (k * d) + (k * d * (d + 1)) / 2
        b[i] = p * np.log(n) - 2 * lkhd

        if b[i] < best_bic:
            best_bic = b[i]
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, l, b
