#!/usr/bin/env python3
"""Converts selected DataFrame values to a NumPy array"""

import numpy as np


def array(df):
    return df[["High", "Close"]].tail(10).to_numpy()
