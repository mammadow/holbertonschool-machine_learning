#!/usr/bin/env python3
"""calculates the cost of a neural network with L2 regularization"""


import tensorflow as tf


def l2_reg_cost(cost, model):
    """calculates the cost of a neural network with L2 regularization"""
    return cost + tf.reduce_sum(model.losses)
