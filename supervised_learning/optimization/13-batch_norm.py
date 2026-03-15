#!/usr/bin/env python3
"""Normalizes an unactivated output using batch normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Applies batch normalization"""
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)

    Z_norm = (Z - mean) / np.sqrt(var + epsilon)

    return gamma * Z_norm + beta
