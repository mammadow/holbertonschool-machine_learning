#!/usr/bin/env python3
"""Module for calculating specificity"""
import numpy as np


def specificity(confusion):
    """Calculates specificity"""
    tp = np.diag(confusion)
    row_sums = np.sum(confusion, axis=1)
    col_sums = np.sum(confusion, axis=0)
    total = np.sum(confusion)

    fp = col_sums - tp
    fn = row_sums - tp
    tn = total - tp - fp - fn

    return tn / (tn + fp)
