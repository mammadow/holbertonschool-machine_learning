#!/usr/bin/env python3
"""Performs agglomerative clustering with Ward linkage"""
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering and returns cluster labels"""
    Z = sch.linkage(X, method='ward')

    plt.figure()
    sch.dendrogram(Z, color_threshold=dist)
    plt.show()

    clss = sch.fcluster(Z, t=dist, criterion='distance') - 1

    return clss
