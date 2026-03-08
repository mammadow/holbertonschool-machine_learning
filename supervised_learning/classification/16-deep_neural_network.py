#!/usr/bin/env python3
"""Module that defines a deep neural network for binary classification."""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Initialize the deep neural network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(nodes, int) and nodes > 0 for nodes in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i, nodes in enumerate(layers):
            layer_num = i + 1
            prev_nodes = nx if i == 0 else layers[i - 1]

            self.weights["W{}".format(layer_num)] = (
                    np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
                    )
            self.weights["b{}".format(layer_num)] = np.zeros((nodes, 1))
