#!/usr/bin/env python3
"""Performs the expectation-maximization algorithm for a GMM"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the EM algorithm for a GMM"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    l = None

    for i in range(iterations):
        g, new_l = expectation(X, pi, m, S)

        pi, m, S = maximization(X, g)

        if verbose and (i == 0 or i % 10 == 0):
            print("Log Likelihood after {} iterations: {:.5f}".format(i, new_l))

        if l is not None and abs(new_l - l) <= tol:
            l = new_l
            break

        l = new_l

    if verbose:
        print("Log Likelihood after {} iterations: {:.5f}".format(i + 1, l))

    return pi, m, S, g, l
