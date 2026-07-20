#!/usr/bin/env python3
"""Create masks."""

import tensorflow as tf


def create_masks(inputs, target):
    """Create masks."""
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((tf.shape(target)[1], tf.shape(target)[1])),
        -1,
        0
    )

    target_padding_mask = tf.cast(
        tf.math.equal(target, 0),
        tf.float32
    )
    target_padding_mask = target_padding_mask[
        :, tf.newaxis, tf.newaxis, :
    ]

    combined_mask = tf.maximum(
        target_padding_mask,
        look_ahead_mask
    )

    return encoder_mask, combined_mask, decoder_mask
