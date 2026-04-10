#!/usr/bin/env python3
"""Flip image module"""

import tensorflow as tf


def flip_image(image):
    """Flip image horizontally"""
    return tf.image.flip_left_right(image)
