#!/usr/bin/env python3
"""Brightness module"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly change image brightness"""
    return tf.image.random_brightness(image, max_delta)
