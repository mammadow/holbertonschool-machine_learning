#!/usr/bin/env python3
"""Normal distribution module."""


class Normal:
    """Represents a normal distribution."""

    pi = 3.1415926535897932
    e = 2.7182818284590452

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize a Normal distribution."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            n = len(data)
            mu = sum(data) / n
            var = sum((x - mu) ** 2 for x in data) / n
            self.mean = float(mu)
            self.stddev = float(var ** 0.5)

    def z_score(self, x):
        """Calculate the z-score of a given x-value."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate the x-value of a given z-score."""
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """Calculate the value of the PDF for a given x-value."""
        coefficient = 1 / (self.stddev * (2 * Normal.pi) ** 0.5)
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return coefficient * (Normal.e ** exponent)

    def cdf(self, x):
        """Calculate the value of the CDF for a given x-value."""
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = z - (z ** 3) / 3 + (z ** 5) / 10 - (z ** 7) / 42 + (z ** 9) / 216
        erf *= 2 / (Normal.pi ** 0.5)
        return 0.5 * (1 + erf)
