#!/usr/bin/env python3
"""Rotate image module"""

import tensorflow as tf


def rotate_image(image):
    """Rotate image 90 degrees CCW"""
    return tf.image.rot90(image)
