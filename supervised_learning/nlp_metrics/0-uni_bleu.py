#!/usr/bin/env python3
"""Unigram BLEU score."""

from collections import Counter
from math import exp


def uni_bleu(references, sentence):
    """Calculate the unigram BLEU score."""
    if len(sentence) == 0:
        return 0

    candidate = Counter(sentence)
    clipped = 0

    for word, count in candidate.items():
        maximum = max(Counter(ref)[word] for ref in references)
        clipped += min(count, maximum)

    precision = clipped / len(sentence)

    r = min([len(ref) for ref in references],
            key=lambda x: (abs(x - len(sentence)), x))
    c = len(sentence)

    bp = 1 if c > r else exp(1 - r / c)

    return bp * precision
