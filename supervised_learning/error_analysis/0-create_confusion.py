#!/usr/bin/env python3
"""Module for creating a confusion matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Create a confusion matrix.
    """
    return np.matmul(labels.T, logits)
