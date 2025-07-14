import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load returns
df = pd.read_csv("strategy_returns.csv", index_col=0, parse_dates=True)
df = df.fillna(0)

mean_returns = df.mean()
cov_matrix = df.cov()
n_assets = len(mean_returns)

# Equal-weighted portfolio
equal_weights = np.ones(n_assets) / n_assets
equal_weight_returns = df.dot(equal_weights)

# Volatility-weighted portfolio
vols = df.std()
inv_vol = 1 / vols
vol_weights = inv_vol / inv_vol.sum()
vol_weighted_returns = df.dot(vol_weights)

# Sharpe-optimized portfolio
def neg_sharpe(weights):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -port_return / port_vol if port_vol > 0 else 1e6

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(n_assets))
result = minimize(neg_sharpe, equal_weights, bounds=bounds, constraints=constraints)
sharpe_weights = result.x
sharpe_opt_returns = df.dot(sharpe_weights)

# Cumulative returns
cum_equal = (1 + equal_weight_returns).cumprod()
cum_vol = (1 + vol_weighted_returns).cumprod()
cum_sharpe = (1 + sharpe_opt_returns).cumprod()

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cum_equal, label="Equal Weight")
plt.plot(cum_vol, label="Volatility Weight")
plt.plot(cum_sharpe, label="Sharpe-Optimized")
plt.title("Cumulative Portfolio Return: Equal vs. Volatility vs. Sharpe-Optimized")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print stats
def print_stats(name, returns, weights=None):
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    print(f"{name} Portfolio:")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2%}")
    print(f"  Volatility: {returns.std():.5f}")
    if weights is not None:
        print(f"  Weights: {np.round(weights, 3)}")
    print("")

print_stats("Equal Weight", equal_weight_returns, equal_weights)
print_stats("Volatility Weight", vol_weighted_returns, vol_weights)
print_stats("Sharpe-Optimized", sharpe_opt_returns, sharpe_weights) 