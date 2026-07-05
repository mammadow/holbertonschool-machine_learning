#!/usr/bin/env python3
"""Cumulative BLEU score."""

from collections import Counter
from math import exp, log


def cumulative_bleu(references, sentence, n):
    """Calculate the cumulative n-gram BLEU score."""
    if len(sentence) == 0:
        return 0

    precisions = []

    for k in range(1, n + 1):
        if len(sentence) < k:
            return 0

        candidate = Counter(
            tuple(sentence[i:i + k])
            for i in range(len(sentence) - k + 1)
        )

        clipped = 0

        for gram, count in candidate.items():
            maximum = max(
                Counter(
                    tuple(ref[i:i + k])
                    for i in range(len(ref) - k + 1)
                )[gram]
                for ref in references
            )
            clipped += min(count, maximum)

        precision = clipped / (len(sentence) - k + 1)

        if precision == 0:
            return 0

        precisions.append(log(precision))

    r = min(
        [len(ref) for ref in references],
        key=lambda x: (abs(x - len(sentence)), x)
    )

    c = len(sentence)

    bp = 1 if c > r else exp(1 - r / c)

    return bp * exp(sum(precisions) / n)
