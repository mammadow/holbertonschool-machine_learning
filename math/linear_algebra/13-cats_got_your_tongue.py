#!/usr/bin/env python3
"""concat"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two numpy arrays along a given axis"""
    return np.concatenate((mat1, mat2), axis=axis)
