#!/usr/bin/env python3
"""Multi-reference Question Answering"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def semantic_search(corpus_path, sentence, model):
    """Performs semantic search on a corpus of documents"""
    documents = [sentence]
    filenames = []

    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
        filepath = os.path.join(corpus_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            documents.append(f.read())
        filenames.append(filename)

    embeddings = model(documents)

    correlation = np.inner(embeddings[0], embeddings[1:])
    closest = np.argmax(correlation)

    filepath = os.path.join(corpus_path, filenames[closest])
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def question_answer_single(question, reference, tokenizer, qa_model):
    """Finds a snippet of text within a reference document to answer a question"""
    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(reference)

    tokens = (
        ['[CLS]'] + question_tokens + ['[SEP]'] +
        paragraph_tokens + ['[SEP]']
    )

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = (
        [0] * (1 + len(question_tokens) + 1) +
        [1] * (len(paragraph_tokens) + 1)
    )

    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids)
    )

    outputs = qa_model([input_word_ids, input_mask, input_type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if not answer or short_end < short_start:
        return None

    return answer


def question_answer(corpus_path):
    """Answers questions from multiple reference texts"""
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    qa_model = hub.load(
        'https://tfhub.dev/see--/bert-uncased-tf2-qa/1')
    se_model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5')

    exit_words = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        question = input('Q: ')
        if question.lower() in exit_words:
            print('A: Goodbye')
            break
        reference = semantic_search(corpus_path, question, se_model)
        answer = question_answer_single(
            question, reference, tokenizer, qa_model)
        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: {}'.format(answer))
