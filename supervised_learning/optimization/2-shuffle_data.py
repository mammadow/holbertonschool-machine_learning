#!/usr/bin/env python3
"""Shuffles two matrices the same way"""

import numpy as np


def shuffle_data(X, Y):
    """Shuffles two matrices the same way"""
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
