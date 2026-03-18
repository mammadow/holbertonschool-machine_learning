#!/usr/bin/env python3
"""calculates the cost of a neural network with L2 regularization"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates the cost of a neural network with L2 regularization"""
    l2_sum = 0

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        l2_sum += np.sum(np.square(W))

    return cost + (lambtha / (2 * m)) * l2_sum
