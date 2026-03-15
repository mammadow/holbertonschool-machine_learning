#!/usr/bin/env python3
"""Calculates normalization constants of a matrix"""

import numpy as np


def normalization_constants(X):
    """Calculates the normalization constants of a matrix"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
