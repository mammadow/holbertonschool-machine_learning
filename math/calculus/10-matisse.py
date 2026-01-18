#!/usr/bin/env python3
"""Module for polynomial derivative"""


def poly_derivative(poly):
    """Calculates the derivatives of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None
    
    derivative = []

    for power in range(1, len(poly)):
        derivative.append(poly[power] * power)
    
    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()
    
    return derivative
