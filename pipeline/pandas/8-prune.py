#!/usr/bin/env python3
"""
function def prune(df):
"""


def prune(df):
    """
    function def prune(df): that takes a pd.DataFrame
    """
    return df.dropna(subset="Close")
