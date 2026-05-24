#!/usr/bin/env python3
"""Fits a Gaussian Mixture Model using sklearn"""
import numpy as np
from sklearn.mixture import GaussianMixture


def gmm(X, k):
    """Fits a GMM and returns parameters and predictions"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None

    model = GaussianMixture(n_components=k, covariance_type='full')
    model.fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
