#!/usr/bin/env python3
"""Deep neural network for binary classification."""
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Initialize the network."""
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
        """Calculates forward propagation."""
        self.__cache["A0"] = X

        for i in range(self.__L):
            W = self.__weights["W{}".format(i + 1)]
            b = self.__weights["b{}".format(i + 1)]
            A_prev = self.__cache["A{}".format(i)]

            Z = np.matmul(W, A_prev) + b
            self.__cache["A{}".format(i + 1)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost using logistic regression."""
        m = Y.shape[1]
        return -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions."""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the network."""
        m = Y.shape[1]
        weights_copy = self.__weights.copy()

        for i in range(self.__L, 0, -1):
            A_curr = cache["A{}".format(i)]
            A_prev = cache["A{}".format(i - 1)]

            if i == self.__L:
                dZ = A_curr - Y
            else:
                W_next = weights_copy["W{}".format(i + 1)]
                dZ = np.matmul(W_next.T, dZ) * A_curr * (1 - A_curr)

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights["W{}".format(i)] = (
                weights_copy["W{}".format(i)] - alpha * dW
            )
            self.__weights["b{}".format(i)] = (
                weights_copy["b{}".format(i)] - alpha * db
            )

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if i % step == 0 or i == iterations:
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    steps.append(i)
                    costs.append(cost)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
