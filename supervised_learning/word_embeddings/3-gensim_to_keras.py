#!/usr/bin/env python3
"""Gensim to Keras."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a gensim model to a Keras embedding layer."""
    keyed_vectors = model.wv
    weights = keyed_vectors.vectors

    return tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )
