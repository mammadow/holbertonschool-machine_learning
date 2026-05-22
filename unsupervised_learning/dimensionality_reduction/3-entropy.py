#!/usr/bin/env python3
"""Computes Shannon entropy and P affinities for t-SNE."""
import numpy as np


def HP(Di, beta):
    """Calculates Shannon entropy and P affinities."""
    P = np.exp(-Di * beta)
    sumP = np.sum(P)
    P = P / sumP
    Hi = -np.sum(P * np.log2(P + 1e-10))
    return Hi, P
