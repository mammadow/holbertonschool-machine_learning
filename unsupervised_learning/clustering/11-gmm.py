#!/usr/bin/env python3
"""Fits a Gaussian Mixture Model using sklearn"""
import sklearn.mixture


def gmm(X, k):
    """Fits a GMM and returns parameters and predictions"""
    if X is None or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None

    model = sklearn.mixture.GaussianMixture(
        n_components=k,
        covariance_type='full'
    )
    model.fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
