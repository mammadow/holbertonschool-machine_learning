#!/usr/bin/env python3
"""Bag of Words embedding module."""

import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Create a Bag of Words embedding matrix."""

    processed = []

    for sentence in sentences:
        sentence = re.sub(r"[^a-zA-Z0-9\s]", "", sentence.lower())
        processed.append(sentence.split())

    if vocab is None:
        features = sorted(set(word
                              for sentence in processed
                              for word in sentence))
    else:
        features = sorted(vocab)

    index = {word: i for i, word in enumerate(features)}

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sentence in enumerate(processed):
        for word in sentence:
            if word in index:
                embeddings[i, index[word]] += 1

    return embeddings, np.array(features)
