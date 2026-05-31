#!/usr/bin/env python3
"""
Bayesian Optimization - full optimization loop
"""

import numpy as np
GP = __import__('2-gp').GaussianProcess
from scipy.stats import norm


class BayesianOptimization:
    """
    Bayesian Optimization using Gaussian Process
    """

    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True
    ):
        """
        Constructor
        """

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        self.X_s = np.linspace(
            bounds[0],
            bounds[1],
            ac_samples
        ).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Expected Improvement acquisition function
        """

        mu, var = self.gp.predict(self.X_s)
        sigma = np.sqrt(var)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        Z = np.zeros_like(mu)

        mask = sigma > 0
        Z[mask] = imp[mask] / sigma[mask]

        EI = np.zeros_like(mu)

        EI[mask] = (
            imp[mask] * norm.cdf(Z[mask]) +
            sigma[mask] * norm.pdf(Z[mask])
        )

        X_next = self.X_s[np.argmax(EI)].reshape(1,)

        return X_next, EI.reshape(-1)

    def optimize(self, iterations=100):
        """
        Runs Bayesian Optimization loop
        """

        for _ in range(iterations):

            X_next, _ = self.acquisition()

            # stop if already sampled
            if np.any(np.all(self.gp.X == X_next, axis=1)):
                break

            Y_next = self.f(X_next)

            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        return self.gp.X[idx].reshape(1,), self.gp.Y[idx].reshape(1,)
