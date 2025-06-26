# regime_hmm.py

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def fit_hmm_model(df, n_components=2, threshold=0.6, random_state=42):
    X = df[["return", "volatility_5", "momentum"]].dropna()
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000, random_state=random_state)
    model.fit(X)

    df = df.copy()
    df = df.loc[X.index]  # align
    df["regime"] = model.predict(X)

    means = df.groupby("regime")["return"].mean()
    bull_index = means.idxmax()
    df["bull_prob_hmm"] = model.predict_proba(X)[:, bull_index]
    df["position"] = df["bull_prob_hmm"].apply(lambda p: 1 if p > threshold else 0)
    df["strategy_return"] = df["position"] * df["return"]
    df["cumulative_strategy_return"] = (1 + df["strategy_return"]).cumprod()
    df["cumulative_market_return"] = (1 + df["return"]).cumprod()

    sharpe = df["strategy_return"].mean() / df["strategy_return"].std() * (24 * 2)**0.5
    total_return = df["cumulative_strategy_return"].iloc[-1] - 1
    drawdown = (df["cumulative_strategy_return"] / df["cumulative_strategy_return"].cummax() - 1).min()

    metrics = {
        "sharpe": round(sharpe, 2) if not np.isnan(sharpe) else None,
        "total_return": round(total_return * 100, 2),
        "max_drawdown": round(drawdown * 100, 2)
    }

    return df, metrics

