#!/usr/bin/env python3
"""Crop image module"""

import tensorflow as tf


def crop_image(image, size):
    """Random crop of image"""
    return tf.image.random_crop(image, size)
