#!/usr/bin/env python3
"""Module to integral of the polynomials"""


def poly_integral(poly, C=0):
    if not isinstance(poly, list) or not isinstance(C, (int, float)):
        return None

    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    integrals = []
    integrals.append(C)
    for a in range(0, len(poly)):
        if poly[a] % (a+1) == 0:
            integrals.append(poly[a]//(a+1))
        else:
            integrals.append(poly[a]/(a+1))

    while len(integrals) > 1 and integrals[-1] == 0:
        integrals.pop()

    return integrals
