#!/usr/bin/env python3
"""
Function def flip_switch(df):
"""


def flip_switch(df):
    """
    Function def flip_switch(df): that takes a pd.DataFrame
    """
    return df.sort_index(ascending=False).T
