#!/usr/bin/env python3
"""slice"""


def np_slice(matrix, axes={}):
    """Slices a numpy.ndarray along specified axes"""
    slices = [slice(None)] * matrix.ndim
    for axis, slc in axes.items():
        slices[axis] = slice(*slc)
    return matrix[tuple(slices)]
