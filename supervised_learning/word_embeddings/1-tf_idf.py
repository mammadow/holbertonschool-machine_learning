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

    n_sentences = len(sentences)
    n_features = len(features)

    tf = np.zeros((n_sentences, n_features))
    df = np.zeros(n_features)

    for i, sentence in enumerate(processed):
        counts = {}
        for word in sentence:
            if word in word_to_index:
                counts[word] = counts.get(word, 0) + 1

        total = len(sentence)

        for word, count in counts.items():
            j = word_to_index[word]
            tf[i, j] = count / total
            df[j] += 1

    idf = np.log(n_sentences / df)

    embeddings = tf * idf

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings, np.array(features)
