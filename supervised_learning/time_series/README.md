# Time Series Forecasting — Bitcoin (BTC)

Forecast the **closing price of BTC at the end of the next hour** from
the **previous 24 hours** of data, using an RNN trained on the raw
Coinbase / Bitstamp minute-level datasets.

## Files

| File | Purpose |
|------|---------|
| `preprocess_data.py` | Clean, resample, split and normalise the raw csv; save a `.npz` archive. |
| `forecast_btc.py` | Build a `tf.data` sliding-window pipeline, then train and validate an LSTM. |

## Usage

```bash
./preprocess_data.py path/to/raw_btc.csv      # writes preprocessed.npz
./forecast_btc.py                             # preprocesses + trains
```

`forecast_btc.py` calls `preprocess` itself, so running it is enough;
`preprocess_data.py` can also be run standalone to inspect the output.

## Preprocessing decisions

The project asks five guiding questions; here is how each is answered.

* **Are all data points useful?** No. Bitcoin was thinly traded in the
  early years and those rows are mostly missing, so everything before
  `2017-01-01` is discarded. Remaining gaps are forward-filled, since a
  missing minute most plausibly held the previous price.
* **Are all features useful?** No. `Open`, `High` and `Low` track
  `Close` almost perfectly, and `Volume_(Currency)` is essentially
  `Volume_(BTC)` times the price. We keep `Close`, `Volume_(BTC)` and
  `Weighted_Price`.
* **Should the data be rescaled?** Yes. Prices in the thousands make an
  RNN train poorly, so every feature is standardised (z-score). The
  mean and standard deviation come **only from the training split** to
  avoid leaking future information; the same statistics are then applied
  to the validation and test splits and saved for de-normalisation.
* **Is the current time window relevant?** The minute granularity is
  finer than an hourly forecast needs, so the data is resampled to one
  row per hour (prices first/max/min/last, volumes summed, weighted
  price averaged). This also shrinks the dataset roughly 60×.
* **How to save it?** As a single compressed `numpy` archive
  (`preprocessed.npz`) holding the train / val / test arrays plus the
  normalisation `mean`, `std` and the feature names.

The split is **chronological** (70% train / 20% val / 10% test) because
shuffling a time series before splitting would let the model see the
future during training.

## The data pipeline

`make_dataset` uses `tf.keras.utils.timeseries_dataset_from_array` to
produce a `tf.data.Dataset` of `(sequence, target)` pairs. Each input
spans 24 consecutive hours and its target is the `Close` of the hour
immediately after the window. The dataset is batched and prefetched;
only the training set is shuffled.

## Model

A single `LSTM(64)` summarises the 24-hour window into one state vector,
a `Dropout(0.2)` layer regularises it, and a `Dense(1)` head emits the
scalar forecast. LSTM is chosen over `SimpleRNN` because its gating
avoids the vanishing-gradient problem over a 24-step horizon. The loss
is **mean-squared error (MSE)**, as required; mean absolute error (MAE)
is reported as a more interpretable, same-units metric.

## Requirements

* Python 3.9, `numpy` 1.25.2, `tensorflow` 2.15, `pandas` 2.2.2
* `pycodestyle` 2.11.1 compliant
