#!/usr/bin/env python3
"""Word2Vec model."""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """Create and train a Word2Vec model."""
    return gensim.models.word2vec.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=int(not cbow),
        epochs=epochs,
        seed=seed,
        workers=workers
    )
