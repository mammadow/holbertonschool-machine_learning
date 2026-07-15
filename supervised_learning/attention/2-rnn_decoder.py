#!/usr/bin/env python3
"""RNN Decoder for machine translation."""

import tensorflow as tf

SelfAttention = __import__("1-self_attention").SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decodes target sequences using attention."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the decoder."""
        super().__init__()

        self.attention = SelfAttention(units)

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """Perform one decoding step."""
        context, _ = self.attention(s_prev, hidden_states)

        x = self.embedding(x)

        context = tf.expand_dims(context, axis=1)

        x = tf.concat([context, x], axis=-1)

        output, state = self.gru(
            x,
            initial_state=s_prev
        )

        output = tf.reshape(output, (-1, output.shape[2]))

        y = self.F(output)

        return y, state
