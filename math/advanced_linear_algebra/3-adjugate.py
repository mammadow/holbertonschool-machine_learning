#!/usr/bin/env python3
"""3-adjugate.py"""
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """Compute the adjugate matrix of a non-empty square matrix."""

    cof = cofactor(matrix)
    n = len(cof)

    return [[cof[i][j] for i in range(n)] for j in range(n)]
