import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parameters
train_years = 5  # years in training window
test_years = 1   # years in test window
start_year = 1993
end_year = 2013

# Load returns
df = pd.read_csv("strategy_returns.csv", index_col=0, parse_dates=True)
df = df.fillna(0)
df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]

# Walk-forward
results = []
dates = []
weights_list = []

for train_start in range(start_year, end_year - train_years - test_years + 2):
    train_end = train_start + train_years - 1
    test_start = train_end + 1
    test_end = test_start + test_years - 1
    train = df[(df.index.year >= train_start) & (df.index.year <= train_end)]
    test = df[(df.index.year >= test_start) & (df.index.year <= test_end)]
    if len(train) == 0 or len(test) == 0:
        continue
    mean_returns = train.mean()
    cov_matrix = train.cov()
    n_assets = len(mean_returns)
    def neg_sharpe(weights):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -port_return / port_vol if port_vol > 0 else 1e6
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    res = minimize(neg_sharpe, np.ones(n_assets)/n_assets, bounds=bounds, constraints=constraints)
    weights = res.x
    weights_list.append(weights)
    # Apply weights to test period
    test_returns = test.dot(weights)
    results.append(test_returns)
    dates.append(test.index)

# Concatenate out-of-sample returns
oos_returns = pd.concat(results)
oos_returns.index = pd.to_datetime(oos_returns.index)
oos_returns = oos_returns.sort_index()
cum_oos = (1 + oos_returns).cumprod()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(cum_oos, label="Walk-Forward Out-of-Sample")
plt.title("Walk-Forward Out-of-Sample Portfolio Equity Curve")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Stats
def print_stats(returns):
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    print(f"Walk-Forward Out-of-Sample Portfolio:")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2%}")
    print(f"  Volatility: {returns.std():.5f}")
    print("")

print_stats(oos_returns) 