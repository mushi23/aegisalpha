# regime_gmm.py

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import argparse
import sys

def fit_gmm_model(df, n_components=3, threshold=0.6, random_state=42):
    features = ["return", "volatility_5", "momentum"]
    print("=== Feature Statistics ===")
    print(df[features].describe())
    print("\nCorrelation Matrix:")
    print(df[features].corr())

    X = df[features].dropna()
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", n_init=10, random_state=random_state)
    gmm.fit(X)

    bull_index = gmm.means_[:, 0].argmax()
    probs = gmm.predict_proba(X)

    df = df.copy()
    df = df.loc[X.index]
    predicted_regimes = gmm.predict(X)
    df["regime_gmm"] = predicted_regimes
    df["bull_prob_gmm"] = probs[:, bull_index]
    df["position"] = df["bull_prob_gmm"].apply(lambda p: 1 if p > threshold else 0)
    df["strategy_return"] = df["position"] * df["return"]
    df["cumulative_strategy_return"] = (1 + df["strategy_return"]).cumprod()
    df["cumulative_market_return"] = (1 + df["return"]).cumprod()

    # Regime diagnostics
    print("\n=== Regime Value Counts ===")
    print(df["regime_gmm"].value_counts())
    print("\n=== Bull Probability Stats ===")
    print(df["bull_prob_gmm"].describe())

    # Plot cumulative returns by regime
    df['cum_return'] = (1 + df['return']).cumprod()
    plt.figure(figsize=(12, 5))
    for reg in np.unique(predicted_regimes):
        plt.plot(df[df['regime_gmm'] == reg]['cum_return'], label=f"Regime {reg}")
    plt.legend()
    plt.title("Cumulative Returns by Regime (GMM)")
    plt.savefig(f'gmm_regime_plot_{df["pair"].iloc[0] if "pair" in df.columns else "default"}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    sharpe = df["strategy_return"].mean() / df["strategy_return"].std() * (24 * 2)**0.5
    total_return = df["cumulative_strategy_return"].iloc[-1] - 1
    drawdown = (df["cumulative_strategy_return"] / df["cumulative_strategy_return"].cummax() - 1).min()

    metrics = {
        "sharpe": round(sharpe, 2) if not np.isnan(sharpe) else None,
        "total_return": round(total_return * 100, 2),
        "max_drawdown": round(drawdown * 100, 2)
    }

    # Save regime outputs
    if 'datetime' in df.columns:
        df[['datetime', 'regime_gmm', 'bull_prob_gmm']].to_csv('gmm_regimes.csv', index=False)
    else:
        print("Warning: 'datetime' column not found, not saving gmm_regimes.csv")

    return df, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to input CSV file')
    parser.add_argument('--n_components', type=int, default=3, help='Number of GMM regimes')
    args = parser.parse_args()
    if not args.csv:
        print("Usage: python regime_gmm.py --csv path/to/data.csv [--n_components 3]")
        sys.exit(1)
    df = pd.read_csv(args.csv)
    fit_gmm_model(df, n_components=args.n_components)
