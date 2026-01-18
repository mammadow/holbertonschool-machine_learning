#!/usr/bin/env python3
"""function that calculates i^2 from 1 to n"""

def summation_i_squared(n):
    sum = 0
    for i in range(n+1):
        sum += i**2
    return sum