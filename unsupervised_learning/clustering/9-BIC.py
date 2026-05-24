#!/usr/bin/env python3
"""
Finds the best number of clusters for a GMM using BIC.
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Computes the best k using Bayesian Information Criterion.
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or kmax > X.shape[0]:
        return None, None, None, None
    if kmin > kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    pis = []
    means = []
    covs = []
    logs = []
    bics = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        pis.append(pi)
        means.append(m)
        covs.append(S)
        logs.append(log_l)

        p = (k * d * (d + 1) / 2) + (d * k) + (k - 1)
        bic = p * np.log(n) - 2 * log_l
        bics.append(bic)

    logs = np.array(logs)
    bics = np.array(bics)

    best_idx = np.argmin(bics)

    best_k = kmin + best_idx
    best_result = (pis[best_idx], means[best_idx], covs[best_idx])

    return best_k, best_result, logs, bics
