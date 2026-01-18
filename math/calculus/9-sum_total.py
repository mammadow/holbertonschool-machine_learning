#!/usr/bin/env python3
"""Module for summation_i_squared function"""


def summation_i_squared(n):
    """function that calculates i^2 from 1 to n"""
    if not isinstance(n, int) or n < 1:
        return None
    return n*(n+1)*(2*n+1)/6
