#!/usr/bin/env python3
"""Preprocess the raw Coinbase/Bitstamp BTC datasets for forecasting."""
import numpy as np
import pandas as pd


FEATURES = ['Close', 'Volume_(BTC)', 'Weighted_Price']


def load_raw(path):
    """Load a raw BTC csv file and index it by datetime."""
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    return df.set_index('Timestamp')


def clean(df, start='2017-01-01'):
    """Drop early/illiquid rows and forward-fill the remaining gaps."""
    df = df[df.index >= start]
    return df.ffill().dropna()


def to_hourly(df):
    """Resample minute-level data to an hourly frequency."""
    agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean',
    }
    return df.resample('1h').agg(agg).dropna()


def split(df, train_frac=0.7, val_frac=0.2):
    """Split a dataframe chronologically into train/val/test parts."""
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))
    return df[:n_train], df[n_train:n_val], df[n_val:]


def normalize(train, val, test):
    """Standardise every split using the training mean and std."""
    mean = train.mean()
    std = train.std()
    return ((train - mean) / std,
            (val - mean) / std,
            (test - mean) / std,
            mean, std)


def preprocess(path, out='preprocessed.npz'):
    """Run the full preprocessing pipeline and save the result."""
    df = load_raw(path)
    df = clean(df)
    df = to_hourly(df)
    df = df[FEATURES]
    train, val, test = split(df)
    train, val, test, mean, std = normalize(train, val, test)
    np.savez(
        out,
        train=train.to_numpy(),
        val=val.to_numpy(),
        test=test.to_numpy(),
        mean=mean.to_numpy(),
        std=std.to_numpy(),
        features=np.array(FEATURES),
    )
    return out


if __name__ == '__main__':
    import sys

    raw = sys.argv[1] if len(sys.argv) > 1 else \
        'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    saved = preprocess(raw)
    print('Saved preprocessed data to {}'.format(saved))
