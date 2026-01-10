#!/usr/bin/env python3
"""Renames and formats timestamp data in a pandas DataFrame"""

import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime, converts it to datetime,
    and returns only the Datetime and Close columns.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
