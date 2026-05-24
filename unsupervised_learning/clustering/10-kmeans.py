#!/usr/bin/env python3
"""Performs K-means using sklearn"""
from sklearn.cluster import KMeans


def kmeans(X, k):
    """Applies KMeans clustering from sklearn"""
    if type(k) is not int or k <= 0:
        return None, None

    model = KMeans(n_clusters=k, n_init=10)
    model.fit(X)

    return model.cluster_centers_, model.labels_
