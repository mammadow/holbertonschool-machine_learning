#!/usr/bin/env python3
"""Module for calculating specifity"""
import numpy as np


def specifity(confusion):
    """Calculates specifity"""
    tp = np.diag(confusion)
    row_sums = np.sum(confusion, axis=1)
    col_sums = np.sum(confusion, axis=0)
    total = np.sum(confusion)

    fp = col_sums - tp
    fn = row_sums - tp
    tn = total - tp - fp - fn

    return tn / (tn + fp)
