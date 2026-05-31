#!/usr/bin/env python3
"""Gaussian Process module"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix
        """
        sqdist = (X1 - X2.T) ** 2

        return (self.sigma_f ** 2 *
                np.exp(-sqdist / (2 * self.l ** 2)))
