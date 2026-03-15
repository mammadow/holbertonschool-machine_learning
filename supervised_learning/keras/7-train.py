#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """Returns the History object generated after training the model"""
    callbacks = []

    if early_stopping and validation_data is not None:
        callbacks.append(
            K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
        )

    if learning_rate_decay and validation_data is not None:
        def lr_schedule(epoch):
            """Calculates the updated learning rate"""
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(
            K.callbacks.LearningRateScheduler(
                lr_schedule,
                verbose=1
            )
        )

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )
