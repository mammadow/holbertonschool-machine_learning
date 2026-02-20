#!/usr/bin/env python3
"""Module to calculate precision for each class."""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.
    """
    tp = np.diag(confusion)
    col_sums = np.sum(confusion, axis=0)
    return tp / col_sums
