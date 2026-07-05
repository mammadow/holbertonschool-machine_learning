#!/usr/bin/env python3
"""Bag of Words embedding module."""

import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix."""

    processed = []

    for sentence in sentences:
        sentence = sentence.lower()
        # Remove possessive 's
        sentence = re.sub(r"'s\b", "", sentence)
        # Remove remaining punctuation
        sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
        processed.append(sentence.split())

    if vocab is None:
        features = sorted(set(
            word
            for sentence in processed
            for word in sentence
        ))
    else:
        features = sorted(vocab)

    word_to_index = {
        word: i
        for i, word in enumerate(features)
    }

    embeddings = np.zeros(
        (len(sentences), len(features)),
        dtype=int
    )

    for i, sentence in enumerate(processed):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings, np.array(features)
