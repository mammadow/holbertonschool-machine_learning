#!/usr/bin/env python3
"""
Bayesian Optimization - Acquisition
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
        Calculates Expected Improvement and next sampling point
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
