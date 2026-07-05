#!/usr/bin/env python3
"""TF-IDF embedding module."""

import re
import numpy as np


def preprocess(sentence):
    """Preprocess a sentence."""
    sentence = sentence.lower()
    sentence = re.sub(r"'s\b", "", sentence)
    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
    return sentence.split()


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding matrix."""

    processed = [preprocess(sentence) for sentence in sentences]

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

    s = len(sentences)
    f = len(features)

    embeddings = np.zeros((s, f))

    # ---------- Document Frequency ----------
    df = np.zeros(f)

    for words in processed:
        seen = set(words)
        for word in seen:
            if word in word_to_index:
                df[word_to_index[word]] += 1

    # ---------- IDF ----------
    idf = np.zeros(f)

    for i in range(f):
        if df[i] > 0:
            idf[i] = np.log(s / df[i])

    # ---------- TF-IDF ----------
    for i, words in enumerate(processed):

        total_words = len(words)

        counts = {}

        for word in words:
            if word in word_to_index:
                counts[word] = counts.get(word, 0) + 1

        for word, count in counts.items():
            j = word_to_index[word]
            tf = count / total_words
            embeddings[i, j] = tf * idf[j]

        # L2 Normalize
        norm = np.linalg.norm(embeddings[i])
        if norm != 0:
            embeddings[i] /= norm

    return embeddings, np.array(features)
