# regime_gmm.py

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def fit_gmm_model(df, n_components=2, threshold=0.6, random_state=42):
    X = df[["return", "volatility_5", "momentum"]].dropna()
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", 
                          n_init=10, random_state=random_state)
    gmm.fit(X)

    bull_index = gmm.means_[:, 0].argmax()
    probs = gmm.predict_proba(X)

    df = df.copy()
    df = df.loc[X.index]
    df["bull_prob_gmm"] = probs[:, bull_index]
    df["position"] = df["bull_prob_gmm"].apply(lambda p: 1 if p > threshold else 0)
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
