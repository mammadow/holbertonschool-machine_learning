#!/usr/bin/env python3
"""Poisson distribution module."""


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize a Poisson distribution.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the PMF for a given number of successes (k)."""
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        fact = 1
        for i in range(1, k+1):
            fact *= i

        e = 2.7182818285

        return (e ** (-self.lambtha)) * (self.lambtha ** k) / fact

    def cdf(self, k):
        """Calculates the CDF for a given number of successes (k)."""
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        e = 2.7182818285

        p = e ** (-self.lambtha)
        cdf_val = p

        for i in range(1, k + 1):
            p = p * self.lambtha / i
            cdf_val += p

        return cdf_val
