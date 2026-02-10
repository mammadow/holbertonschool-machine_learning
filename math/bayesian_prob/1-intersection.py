#!/usr/bin/env python3
"""
Bayesian Probability - Intersection (Joint Probability)
"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data given various
    hypothetical probabilities
    """

    binomial_coef = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n - x)
    )

    likelihood_values = binomial_coef * (P ** x) * ((1 - P) ** (n - x))

    return likelihodo_values


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining this data with the various
    hypothetical probabilities
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
                "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    likelihood_values = likelihood(x, n, P)

    intersection_values = likelihood_values * Pr

    return intersection_values
