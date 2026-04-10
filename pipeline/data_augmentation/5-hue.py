#!/usr/bin/env python3
"""Hue module"""

import tensorflow as tf


def change_hue(image, delta):
    """Change image hue"""
    return tf.image.adjust_hue(image, delta)
