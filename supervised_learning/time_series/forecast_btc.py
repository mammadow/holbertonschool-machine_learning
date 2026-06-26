#!/usr/bin/env python3
"""Forecast the next-hour BTC closing price with an LSTM RNN."""
import numpy as np
import tensorflow as tf

from preprocess_data import preprocess


WINDOW = 24
BATCH = 256
CLOSE_IDX = 0


def make_dataset(data, window=WINDOW, batch=BATCH, shuffle=False):
    """Build a sliding-window tf.data.Dataset of (sequence, target)."""
    inputs = data[:-window]
    targets = data[window:, CLOSE_IDX]
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=inputs,
        targets=targets,
        sequence_length=window,
        batch_size=batch,
        shuffle=shuffle,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(n_features, window=WINDOW):
    """Create and compile the LSTM forecasting model with MSE loss."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_features)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main(raw='coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv',
         epochs=20):
    """Preprocess, train and evaluate the forecasting model."""
    path = preprocess(raw)
    data = np.load(path, allow_pickle=True)
    train, val, test = data['train'], data['val'], data['test']

    train_ds = make_dataset(train, shuffle=True)
    val_ds = make_dataset(val)
    test_ds = make_dataset(test)

    model = build_model(train.shape[1])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    loss, mae = model.evaluate(test_ds)
    print('test MSE: {:.4f}  test MAE: {:.4f}'.format(loss, mae))
    return model


if __name__ == '__main__':
    main()
