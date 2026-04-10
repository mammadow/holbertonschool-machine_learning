#!/usr/bin/env python3
"""Contrast module"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly change image contrast"""
    return tf.image.random_contrast(image, lower, upper)
