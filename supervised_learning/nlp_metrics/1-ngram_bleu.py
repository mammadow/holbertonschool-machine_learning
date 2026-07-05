#!/usr/bin/env python3
"""N-gram BLEU score."""

from collections import Counter
from math import exp


def ngram_bleu(references, sentence, n):
    """Calculate the n-gram BLEU score."""
    if len(sentence) < n:
        return 0

    candidate = Counter(
        tuple(sentence[i:i+n])
        for i in range(len(sentence) - n + 1)
    )

    clipped = 0

    for gram, count in candidate.items():
        maximum = max(
            Counter(
                tuple(ref[i:i+n])
                for i in range(len(ref) - n + 1)
            )[gram]
            for ref in references
        )
        clipped += min(count, maximum)

    precision = clipped / (len(sentence) - n + 1)

    r = min(
        [len(ref) for ref in references],
        key=lambda x: (abs(x - len(sentence)), x)
    )
    c = len(sentence)

    bp = 1 if c > r else exp(1 - r / c)

    return bp * precision
