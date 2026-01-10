#!/usr/bin/env python3
"""
function def fill(df):
"""


def fill(df):
    """
    function def fill(df): that takes a pd.DataFrame
    """
    df = df.drop(columns="Weighted_Price")
    df["Close"] = df["Close"].fillna(method="ffill")
    for col in ["Open", "High", "Low"]:
        df[col] = df[col].fillna(df["Close"])
    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[col] = df[col].fillna(0)

    return df
