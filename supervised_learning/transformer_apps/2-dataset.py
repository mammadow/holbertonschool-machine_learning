#!/usr/bin/env python3
"""Dataset class."""

import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Dataset."""

    def __init__(self):
        """Initialize dataset."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Train tokenizers."""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def pt_iterator():
            """PT iterator."""
            for pt, _ in data:
                yield pt.numpy().decode("utf-8")

        def en_iterator():
            """EN iterator."""
            for _, en in data:
                yield en.numpy().decode("utf-8")

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(),
            vocab_size=2 ** 13
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode sentences."""
        pt_tokens = [self.tokenizer_pt.vocab_size]
        pt_tokens.extend(
            self.tokenizer_pt.encode(
                pt.numpy().decode("utf-8"),
                add_special_tokens=False
            )
        )
        pt_tokens.append(self.tokenizer_pt.vocab_size + 1)

        en_tokens = [self.tokenizer_en.vocab_size]
        en_tokens.extend(
            self.tokenizer_en.encode(
                en.numpy().decode("utf-8"),
                add_special_tokens=False
            )
        )
        en_tokens.append(self.tokenizer_en.vocab_size + 1)

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow encode wrapper."""
        pt, en = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )
        pt.set_shape([None])
        en.set_shape([None])
        return pt, en
