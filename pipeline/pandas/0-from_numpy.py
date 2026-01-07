#!/usr/bin/env python3
"""
This is a code for function that creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Function that creates pd.DataFrame from a np.ndarray
    """
    cols = [chr(ord('A') + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=cols)
