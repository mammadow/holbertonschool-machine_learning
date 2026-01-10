#!/usr/bin/env python3
"""Converts selected DataFrame values to a NumPy array"""


def array(df):
    """
    Function that converts selected DataFrame values to a NumPy array
    """
    return df[["High", "Close"]].tail(10).to_numpy()
