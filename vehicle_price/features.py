import pandas as pd

def add_features(df):
    df = df.copy()
    df["brand"] = df["CarName"].str.split().str[0]
    return df
