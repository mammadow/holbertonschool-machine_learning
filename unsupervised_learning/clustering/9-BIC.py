#!/usr/bin/env python3
"""Bayesian Information Criterion for GMM"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find best k using BIC for Gaussian Mixture Model"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0 or kmax > X.shape[0]:
        return None, None, None, None

    if kmax < kmin:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    results_pi = []
    results_m = []
    results_S = []
    log_likelihoods = []
    bic_values = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, l = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        results_pi.append(pi)
        results_m.append(m)
        results_S.append(S)
        log_likelihoods.append(l)

        p = (k * d * (d + 1) / 2) + (d * k) + (k - 1)
        bic = p * np.log(n) - 2 * l
        bic_values.append(bic)

    log_likelihoods = np.array(log_likelihoods)
    bic_values = np.array(bic_values)

    best_idx = np.argmin(bic_values)

    best_k = kmin + best_idx
    best_result = (results_pi[best_idx],
                   results_m[best_idx],
                   results_S[best_idx])

    return best_k, best_result, log_likelihoods, bic_values
