import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load strategy returns
df = pd.read_csv("strategy_returns.csv", index_col=0, parse_dates=True)
df = df.fillna(0)

# Compute inverse volatility weights
daily_vols = df.std()
weights = 1 / daily_vols
weights /= weights.sum()
print("Volatility Weights:")
for pair, w in weights.items():
    print(f"  {pair}: {w:.2%}")

# Compute volatility-weighted portfolio returns
vol_weighted_portfolio = df.dot(weights)
cum_vol_weighted = (1 + vol_weighted_portfolio).cumprod()

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cum_vol_weighted, label="Volatility-Weighted Portfolio")
plt.title("Cumulative Portfolio Return (Volatility-Weighted)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print stats
sharpe = vol_weighted_portfolio.mean() / vol_weighted_portfolio.std() * np.sqrt(252) if vol_weighted_portfolio.std() > 0 else np.nan
max_drawdown = (cum_vol_weighted / cum_vol_weighted.cummax() - 1).min()
print(f"Volatility-Weighted Portfolio Sharpe: {sharpe:.2f}")
print(f"Volatility-Weighted Portfolio Max Drawdown: {max_drawdown:.2%}")
print(f"Volatility-Weighted Portfolio Volatility: {vol_weighted_portfolio.std():.5f}") 