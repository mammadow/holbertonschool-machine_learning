#!/usr/bin/env python3
"""Bag of Words embedding."""

import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix."""
    processed = []

    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r"'s\b", "", sentence)
        sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
        processed.append(sentence.split())

    if vocab is None:
        features = sorted({
            word
            for sentence in processed
            for word in sentence
        })
    else:
        features = list(vocab)

    word_to_index = {
        word: i
        for i, word in enumerate(features)
    }

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sentence in enumerate(processed):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings, np.array(features)
