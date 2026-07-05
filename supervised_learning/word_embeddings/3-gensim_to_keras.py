#!/usr/bin/env python3
"""Module that converts a gensim word2vec model to a keras Embedding layer"""
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a trainable keras Embedding"""
    weights = model.wv.vectors

    return Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True,
    )
