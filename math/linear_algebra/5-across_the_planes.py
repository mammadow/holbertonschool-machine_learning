#!/usr/bin/env python3
"""add_matrices2D"""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise"""
    if len(mat1) != len(mat2):
        return None
    for r in range(len(mat1)):
        if len(mat1[r]) != len(mat2[r]):
            return None
    return [[mat1[r][c] + mat2[r][c] for c in range(len(mat1[0]))]
            for r in range(len(mat1))]
