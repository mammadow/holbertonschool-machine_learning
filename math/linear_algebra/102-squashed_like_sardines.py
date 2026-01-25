#!/usr/bin/env python3
"""cat matrices"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""

    def matrix_shape(mat):
        shape = []
        while isinstance(mat, list):
            shape.append(len(mat))
            mat = mat[0]
        return shape

    def deep_copy(mat):
        if isinstance(mat, list):
            return [deep_copy(x) for x in mat]
        return mat

    def concat(a, b, ax):
        if ax == 0:
            return deep_copy(a) + deep_copy(b)
        return [concat(a[i], b[i], ax - 1) for i in range(len(a))]

    s1 = matrix_shape(mat1)
    s2 = matrix_shape(mat2)

    if len(s1) != len(s2) or axis < 0 or axis >= len(s1):
        return None
    for i in range(len(s1)):
        if i != axis and s1[i] != s2[i]:
            return None

    return concat(mat1, mat2, axis)
