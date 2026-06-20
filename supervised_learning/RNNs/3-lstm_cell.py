#!/usr/bin/env python3
"""LSTM Cell module"""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """Initializes weights and biases for the LSTM cell"""
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step"""
        concat = np.concatenate((h_prev, x_t), axis=1)

        ft = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        ut = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        cct = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        c_next = ft * c_prev + ut * cct

        ot = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)
        h_next = ot * np.tanh(c_next)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, c_next, y

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Softmax activation function"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
