#!/usr/bin/env python3
"""Computes BIC for GMM model selection"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Selects best K using Bayesian Information Criterion"""
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
    best_bic = np.inf
    best_result = None

    for i, k in enumerate(ks):
        try:
            pi, m, S, g, log_l = expectation_maximization(
                X, k, iterations=iterations, tol=tol, verbose=verbose
            )
        except Exception:
            continue

        if log_l is None:
            continue

        l[i] = log_l

        p = k - 1 + k * d + (k * d * (d + 1)) // 2
        bic = p * np.log(n) - 2 * log_l
        b[i] = bic

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    if best_k is None:
        return None, None, None, None

    pi, m, S, g, log_l = expectation_maximization(
        X, best_k, iterations=iterations, tol=tol, verbose=False
    )

    return best_k, best_result, l, b
