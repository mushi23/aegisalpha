import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load returns
df = pd.read_csv("strategy_returns.csv", index_col=0, parse_dates=True)
df = df.fillna(0)

# Equal-weighted portfolio
equal_weight_returns = df.mean(axis=1)
cum_equal = (1 + equal_weight_returns).cumprod()

# Volatility-weighted portfolio
vols = df.std()
inv_vol = 1 / vols
weights = inv_vol / inv_vol.sum()
vol_weighted_returns = df.dot(weights)
cum_vol = (1 + vol_weighted_returns).cumprod()

# Blended portfolio (optional)
blended_returns = 0.5 * equal_weight_returns + 0.5 * vol_weighted_returns
cum_blended = (1 + blended_returns).cumprod()

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cum_equal, label="Equal Weight")
plt.plot(cum_vol, label="Volatility Weight")
plt.plot(cum_blended, label="Blended (50/50)", linestyle='--')
plt.title("Cumulative Portfolio Return: Equal vs. Volatility Weight")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print stats
def print_stats(name, returns):
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    print(f"{name} Portfolio:")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2%}")
    print(f"  Volatility: {returns.std():.5f}")
    print("")

print_stats("Equal Weight", equal_weight_returns)
print_stats("Volatility Weight", vol_weighted_returns)
print_stats("Blended (50/50)", blended_returns) 