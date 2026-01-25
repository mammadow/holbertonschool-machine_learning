#!/usr/bin/env python3
"""add two matrices"""


def add_matrices(mat1, mat2):
    """Adds two matrices of any dimension"""
    if type(mat1) != type(mat2):
        return None

    if isinstance(mat1, list):
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            summed = add_matrices(mat1[i], mat2[i])
            if summed is None:
                return None
            result.append(summed)
        return result

    return mat1 + mat2
