#!/usr/bin/env python3
"""Computes t-SNE cost function."""
import numpy as np


def cost(P, Q):
    """Calculates the KL divergence cost of the t-SNE transformation."""
    eps = 1e-12
    P = np.clip(P, eps, None)
    Q = np.clip(Q, eps, None)

    C = np.sum(P * np.log(P / Q))
    return C
