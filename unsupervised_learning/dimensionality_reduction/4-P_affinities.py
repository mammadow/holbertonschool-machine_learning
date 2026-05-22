#!/usr/bin/env python3
"""Computes symmetric P affinities for t-SNE."""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """Calculates symmetric P affinities using binary search."""
    D, P, betas, H = P_init(X, perplexity)
    n = X.shape[0]

    logU = np.log2(perplexity)

    for i in range(n):
        Di = np.concatenate((D[i, :i], D[i, i+1:]))

        beta_min = None
        beta_max = None
        beta = betas[i, 0]

        H_i, Pi = HP(Di, beta)

        tries = 0
        while abs(H_i - logU) > tol and tries < 50:
            if H_i > logU:
                beta_min = beta
                if beta_max is None:
                    beta *= 2
                else:
                    beta = (beta + beta_max) / 2
            else:
                beta_max = beta
                if beta_min is None:
                    beta /= 2
                else:
                    beta = (beta + beta_min) / 2

            H_i, Pi = HP(Di, beta)
            tries += 1

        P[i, :i] = Pi[:i]
        P[i, i+1:] = Pi[i:]
        P[i, i] = 0

    P = (P + P.T)
    P = P / np.sum(P)

    return P
