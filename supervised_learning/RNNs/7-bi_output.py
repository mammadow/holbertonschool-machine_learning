#!/usr/bin/env python3
"""Bidirectional Cell module"""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """Initializes weights and biases for the bidirectional cell"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculates the hidden state in the forward direction"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """Calculates the hidden state in the backward direction"""
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(concat, self.Whb) + self.bhb)

        return h_prev

    def output(self, H):
        """Calculates all outputs for the RNN"""
        t = H.shape[0]
        Y = []

        for step in range(t):
            y_linear = np.matmul(H[step], self.Wy) + self.by
            y = self.softmax(y_linear)
            Y.append(y)

        Y = np.array(Y)

        return Y

    @staticmethod
    def softmax(x):
        """Softmax activation function"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
