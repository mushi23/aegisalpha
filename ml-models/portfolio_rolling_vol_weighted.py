import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
window = 60  # rolling window size in days

# Load strategy returns
df = pd.read_csv("strategy_returns.csv", index_col=0, parse_dates=True)
df = df.fillna(0)

# Compute rolling inverse volatility weights
def rolling_vol_weights(returns, window):
    # rolling std for each pair
    rolling_vol = returns.rolling(window).std()
    # inverse vol, normalized
    inv_vol = 1 / rolling_vol
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return weights

weights = rolling_vol_weights(df, window)
# Align weights and returns
weights = weights.shift(1).fillna(0)  # Use yesterday's weights for today's returns

# Compute rolling-vol-weighted portfolio returns
rolling_vol_portfolio = (df * weights).sum(axis=1)
cum_rolling_vol = (1 + rolling_vol_portfolio).cumprod()

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cum_rolling_vol, label=f"Rolling {window}-Day Volatility-Weighted Portfolio")
plt.title(f"Cumulative Portfolio Return (Rolling {window}-Day Volatility-Weighted)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print stats
sharpe = rolling_vol_portfolio.mean() / rolling_vol_portfolio.std() * np.sqrt(252) if rolling_vol_portfolio.std() > 0 else np.nan
max_drawdown = (cum_rolling_vol / cum_rolling_vol.cummax() - 1).min()
print(f"Rolling Volatility-Weighted Portfolio Sharpe: {sharpe:.2f}")
print(f"Rolling Volatility-Weighted Portfolio Max Drawdown: {max_drawdown:.2%}")
print(f"Rolling Volatility-Weighted Portfolio Volatility: {rolling_vol_portfolio.std():.5f}") 