#!/usr/bin/env python3
"""Gaussian Process module"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian Process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize Gaussian Process"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """RBF covariance kernel"""
        sqdist = (X1 - X2.T) ** 2
        return self.sigma_f ** 2 * np.exp(-sqdist / (2 * self.l ** 2))

    def predict(self, X_s):
        """Predict mean and variance"""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        K_inv_Y = np.linalg.solve(self.K, self.Y)
        mu = K_s.T @ K_inv_Y

        v = np.linalg.solve(self.K, K_s)
        cov = K_ss - K_s.T @ v

        return mu.reshape(-1), np.diag(cov)

    def update(self, X_new, Y_new):
        """Update Gaussian Process with new sample"""
        X_new = X_new.reshape(-1, 1)
        Y_new = Y_new.reshape(-1, 1)

        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
