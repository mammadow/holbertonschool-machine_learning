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
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights["W1"] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                )
            else:
                self.__weights["W{}".format(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1]) *
                    np.sqrt(2 / layers[i - 1])
                )

            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for L."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.__cache["A0"] = X

        for i in range(self.__L):
            W = self.__weights["W{}".format(i + 1)]
            b = self.__weights["b{}".format(i + 1)]
            A_prev = self.__cache["A{}".format(i)]

            Z = np.matmul(W, A_prev) + b
            self.__cache["A{}".format(i + 1)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A{}".format(self.__L)], self.__cache
