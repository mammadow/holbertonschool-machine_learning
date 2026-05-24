#!/usr/bin/env python3
"""Bayesian Information Criterion for Gaussian Mixture Models"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using BIC"""

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
    all_lkhds = []
    all_bs = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, lkhd = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None or m is None or S is None:
            return None, None, None, None

        all_pis.append(pi)
        all_ms.append(m)
        all_Ss.append(S)
        all_lkhds.append(lkhd)

        p = (k * d * (d + 1) / 2) + (d * k) + (k - 1)
        all_bs.append(p * np.log(n) - 2 * lkhd)

    all_lkhds = np.array(all_lkhds)
    all_bs = np.array(all_bs)

    best_k = np.argmin(all_bs)
    best_result = (all_pis[best_k], all_ms[best_k], all_Ss[best_k])

    return best_k + kmin, best_result, all_lkhds, all_bs
