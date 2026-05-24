#!/usr/bin/env python3
"""Bayesian Information Criterion for GMM"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find best number of clusters using BIC"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] <= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    all_pis = []
    all_ms = []
    all_Ss = []
    all_l = []
    all_b = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, l = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None or m is None or S is None:
            return None, None, None, None

        all_pis.append(pi)
        all_ms.append(m)
        all_Ss.append(S)
        all_l.append(l)

        p = (k * d * (d + 1) / 2) + (d * k) + (k - 1)
        all_b.append(p * np.log(n) - 2 * l)

    all_l = np.array(all_l)
    all_b = np.array(all_b)

    best_k = int(np.argmin(all_b))

    return best_k + kmin, (all_pis[best_k], all_ms[best_k], all_Ss[best_k]), all_l, all_b
