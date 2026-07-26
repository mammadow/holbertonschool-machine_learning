#!/usr/bin/env python3
"""Semantic search on a corpus of documents"""
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Performs semantic search on a corpus of documents"""
    model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5')

    documents = [sentence]
    filenames = []

    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
        filepath = os.path.join(corpus_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            documents.append(f.read())
        filenames.append(filename)

    embeddings = model(documents)

    correlation = np.inner(embeddings[0], embeddings[1:])
    closest = np.argmax(correlation)

    filepath = os.path.join(corpus_path, filenames[closest])
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()
