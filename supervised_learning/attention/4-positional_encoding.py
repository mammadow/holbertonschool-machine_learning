#!/usr/bin/env python3
"""Positional encoding."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculate the positional encoding."""
    position = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)

    angles = position / np.power(
        10000,
        (2 * (i // 2)) / dm
    )

    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return angles
