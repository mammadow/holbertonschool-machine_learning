#!/usr/bin/env python3
"""GRU Cell module."""

import numpy as np


class GRUCell:
    """Represents a GRU cell."""

    def __init__(self, i, h, o):
        """Initialize the GRU cell."""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """Compute sigmoid activation."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Compute softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r = self.sigmoid(np.matmul(concat, self.Wr) + self.br)

        concat_h = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.matmul(concat_h, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_hat

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
