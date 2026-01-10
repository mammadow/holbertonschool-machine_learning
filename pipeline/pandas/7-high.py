#!/usr/bin/env python3
"""
function def high(df):
"""


def high(df):
    """
    function def high(df): that takes a pd.DataFrame
    """
    return df.sort_values(by="High", ascending=False)
