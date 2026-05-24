#!/usr/bin/env python3
"""Finds best number of clusters for a GMM using BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Performs model selection using Bayesian Information Criterion"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    if kmin < 1 or kmin > kmax:
        return None, None, None, None

    ks = np.arange(kmin, kmax + 1)
    l = np.zeros(len(ks))
    b = np.zeros(len(ks))

    best_k = None
    best_l = None
    best_result = None

    for i, k in enumerate(ks):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose
        )

        l[i] = log_l

        p = (k - 1) + (k * d) + (k * d * (d + 1)) // 2
        b[i] = p * np.log(n) - 2 * log_l

        if best_l is None or b[i] < b[ks == best_k][0] if best_k is not None else True:
            best_k = k
            best_l = log_l
            best_result = (pi, m, S)

    best_idx = np.argmin(b)
    best_k = ks[best_idx]
    best_result = expectation_maximization(
        X, best_k, iterations=iterations, tol=tol, verbose=False
    )[:3]

    return best_k, best_result, l, b
