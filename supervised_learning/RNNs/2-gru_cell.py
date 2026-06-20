#!/usr/bin/env python3
"""GRU Cell implementation."""

import numpy as np


class GRUCell:
    """Represents a GRU cell."""

    def __init__(self, i, h, o):
        """Initialize weights and biases."""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = self._sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r = self._sigmoid(np.matmul(concat, self.Wr) + self.br)

        r_h = r * h_prev
        concat_candidate = np.concatenate((r_h, x_t), axis=1)

        h_tilde = np.tanh(np.matmul(concat_candidate, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde

        y = self._softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y

    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """Softmax activation function."""
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
