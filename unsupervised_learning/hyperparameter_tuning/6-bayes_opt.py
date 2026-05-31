#!/usr/bin/env python3
"""
Bayesian optimization of neural network hyperparameters using GPyOpt
"""

import numpy as np
import GPyOpt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os


# ----------------------------
# Data (simple dataset example)
# ----------------------------
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_val = x_val.reshape(-1, 28 * 28) / 255.0


# ----------------------------
# Model builder
# ----------------------------
def build_model(lr, units, dropout, l2_reg):
    """Create a simple feedforward network"""
    model = keras.Sequential([
        layers.Dense(int(units), activation="relu",
                     kernel_regularizer=keras.regularizers.l2(l2_reg),
                     input_shape=(784,)),
        layers.Dropout(dropout),
        layers.Dense(10, activation="softmax")
    ])

    opt = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ----------------------------
# Objective function for BO
# ----------------------------
def objective(params):
    """GPyOpt objective function (minimize validation loss)"""

    lr = float(params[0][0])
    units = int(params[0][1])
    dropout = float(params[0][2])
    l2_reg = float(params[0][3])
    batch_size = int(params[0][4])

    model = build_model(lr, units, dropout, l2_reg)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )

    val_loss = history.history["val_loss"][-1]

    # Save best model checkpoint
    filename = f"model_lr{lr:.5f}_u{units}_d{dropout:.2f}_l2{l2_reg:.5f}_bs{batch_size}.keras"
    model.save(filename)

    return val_loss


# ----------------------------
# Search space (5 hyperparams)
# ----------------------------
domain = [
    {"name": "lr", "type": "continuous", "domain": (1e-5, 1e-2)},
    {"name": "units", "type": "discrete", "domain": (32, 64, 128, 256)},
    {"name": "dropout", "type": "continuous", "domain": (0.0, 0.5)},
    {"name": "l2", "type": "continuous", "domain": (1e-6, 1e-2)},
    {"name": "batch_size", "type": "discrete", "domain": (32, 64, 128)},
]


# ----------------------------
# Run Bayesian Optimization
# ----------------------------
optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective,
    domain=domain,
    acquisition_type="EI"
)

optimizer.run_optimization(max_iter=30)


# ----------------------------
# Plot convergence
# ----------------------------
optimizer.plot_convergence()
plt.savefig("convergence.png")


# ----------------------------
# Save report
# ----------------------------
best_x = optimizer.X[np.argmin(optimizer.Y)]
best_y = np.min(optimizer.Y)

report = f"""
Bayesian Optimization Report
============================

Best validation loss: {best_y}

Best hyperparameters:
- learning rate: {best_x[0]}
- units: {best_x[1]}
- dropout: {best_x[2]}
- l2: {best_x[3]}
- batch size: {best_x[4]}

Total iterations: {len(optimizer.Y)}
"""

with open("bayes_opt.txt", "w") as f:
    f.write(report)
