#!/usr/bin/env python3
"""updates the weights and biases"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases of a neural network"""
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)].copy()

        dW = (np.matmul(dZ, A_prev.T) / m) + ((lambtha / m) * W)
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dZ = np.matmul(W.T, dZ) *
            (1 - np.power(cache["A" + str(i - 1)], 2))

        weights["W" + str(i)] = weights["W" + str(i)] - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db
