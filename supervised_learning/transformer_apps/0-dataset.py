#!/usr/bin/env python3
"""Dataset class."""

from setup import load_pt2en
import transformers


class Dataset:
    """Dataset."""

    def __init__(self):
        """Initialize dataset."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Train tokenizers."""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            (pt.numpy().decode("utf-8") for pt, _ in data),
            vocab_size=2 ** 13
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            (en.numpy().decode("utf-8") for _, en in data),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
