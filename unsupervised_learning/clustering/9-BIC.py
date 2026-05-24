#!/usr/bin/env python3
"""Bayesian Information Criterion for selecting the best number of
clusters in a Gaussian Mixture Model."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < 1):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n
    if kmax <= kmin:
        return None, None, None, None

    k_range = kmax - kmin + 1
    log_likelihoods = np.empty(k_range)
    bics = np.empty(k_range)
    results = []

    for i in range(k_range):
        k = kmin + i
        pi, m, S, _, ll = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None
        results.append((pi, m, S))
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)
        log_likelihoods[i] = ll
        bics[i] = p * np.log(n) - 2 * ll

    best_idx = int(np.argmin(bics))
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, log_likelihoods, bics
