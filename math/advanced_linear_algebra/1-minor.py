#!/usr/bin/env python3
"""1-minor.py"""


def _determinant(mat):
    """Compute determinant for internal use."""
    if mat == [[]]:
        return 1

    n = len(mat)

    if n == 1:
        return mat[0][0]

    if n == 2:
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

    det = 0
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in mat[1:]]
        det += ((-1) ** j) * mat[0][j] * _determinant(minor)
    return det


def minor(matrix):
    """Compute the minor matrix of a non-empty square matrix."""

    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or matrix == [[]] or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
    return [[1]]

    minors = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            sub = [r[:j] + r[j+1:] for k, r in enumerate(matrix) if k != i]
            row_minors.append(_determinant(sub))
        minors.append(row_minors)

    return minors
