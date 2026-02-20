#!/usr/bin/env python3
"""Module for calculating sensitivity from a confusion matrix."""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.
    """
    tp = np.diag(confusion)
    row_sums = np.sum(confusion, axis=1)
    return tp / row_sums
