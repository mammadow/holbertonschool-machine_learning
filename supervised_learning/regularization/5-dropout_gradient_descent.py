#!/usr/bin/env python3
"""updates the weights of a neural network with Dropout regularization
using gradient descent"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates the weights of a neural network with Dropout regularization
    using gradient descent"""
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)].copy()

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W" + str(i)] -= alpha * dW
        weights["b" + str(i)] -= alpha * db

        if i > 1:
            dA = np.matmul(W.T, dZ)
            dA *= cache["D" + str(i - 1)]
            dA /= keep_prob
            dZ = dA * (1 - np.power(cache["A" + str(i - 1)], 2))
