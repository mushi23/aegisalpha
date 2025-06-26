import pandas as pd

def generate_features(df):
    df["return"] = df["close"].pct_change().fillna(0)
    df["volatility_5"] = df["return"].rolling(5).std().fillna(0)
    df["momentum"] = df["close"] - df["close"].shift(5)
    df["momentum"] = df["momentum"].fillna(0)
    return df
