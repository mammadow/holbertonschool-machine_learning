#!/usr/bin/env python3
"""Updates the learning rate using inverse time decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Calculates the decayed learning rate"""
    return alpha / (1 + decay_rate * (global_step // decay_step))
