#!/usr/bin/env python3
"""LeNet-5 model."""

from tensorflow import keras as K


def lenet5(X):
    """Build and compile a modified LeNet-5 model."""
    initializer = K.initializers.he_normal(seed=0)

    A = K.layers.Conv2D(filters=6,
                        kernel_size=(5, 5),
                        padding='same',
                        activation='relu',
                        kernel_initializer=initializer)(X)
    A = K.layers.MaxPool2D(pool_size=(2, 2),
                           strides=(2, 2))(A)

    A = K.layers.Conv2D(filters=16,
                        kernel_size=(5, 5),
                        padding='valid',
                        activation='relu',
                        kernel_initializer=initializer)(A)
    A = K.layers.MaxPool2D(pool_size=(2, 2),
                           strides=(2, 2))(A)

    A = K.layers.Flatten()(A)

    A = K.layers.Dense(units=120,
                       activation='relu',
                       kernel_initializer=initializer)(A)
    A = K.layers.Dense(units=84,
                       activation='relu',
                       kernel_initializer=initializer)(A)
    Y = K.layers.Dense(units=10,
                       activation='softmax',
                       kernel_initializer=initializer)(A)

    model = K.Model(inputs=X, outputs=Y)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
