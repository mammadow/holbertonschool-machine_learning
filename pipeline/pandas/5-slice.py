#!/usr/bin/env python3
"""
Function def slice(df): that takes a pd.DataFrame
"""

def slice(df):
    """
    Function def slice(df): that takes a pd.DataFrame
    """
    return df[["High", "Low", "Close", "Volume_BTC"]][::60]
