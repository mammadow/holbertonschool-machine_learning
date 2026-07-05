#!/usr/bin/env python3
"""TF-IDF embedding."""

import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding matrix."""
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

    embeddings = np.zeros((len(sentences), len(features)))
    df = np.zeros(len(features))

    for i, sentence in enumerate(processed):
        seen = set()

        for word in sentence:
            if word in word_to_index:
                j = word_to_index[word]
                embeddings[i, j] += 1
                seen.add(word)

        for word in seen:
            df[word_to_index[word]] += 1

    idf = np.log(len(sentences) / df)

    embeddings *= idf

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1

    embeddings /= norms

    return embeddings, np.array(features)
