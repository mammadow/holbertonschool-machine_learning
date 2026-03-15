#!/usr/bin/env python3
"""Calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set"""
    v = 0
    avg = []

    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta ** t)
        avg.append(v_corrected)

    return avg
