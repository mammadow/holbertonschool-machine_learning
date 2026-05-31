#!/usr/bin/env python3
"""
Gaussian Process module
"""

import numpy as np


class GaussianProcess:
    """Noiseless 1D Gaussian Process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize Gaussian Process"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """RBF kernel"""
        first = np.sum(X1 ** 2, 1).reshape(-1, 1)
        second = np.sum(X2 ** 2, 1)
        sqdist = first + second - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """Predict mean and variance"""
        K = self.kernel(self.X, self.X)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        K_inv = np.linalg.inv(K)

        mu = K_s.T @ K_inv @ self.Y
        cov = K_ss - K_s.T @ K_inv @ K_s

        mu = mu.reshape(X_s.shape[0])
        sigma = np.diagonal(cov)

        return mu, sigma

    def update(self, X_new, Y_new):
        """Update GP with new sample"""
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
