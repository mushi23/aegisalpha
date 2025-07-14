import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parameters
window_years = 5   # rolling window size for optimization
rebalance_months = 12  # rebalance frequency in months
start_year = 1993
end_year = 2013

# Load returns
df = pd.read_csv("strategy_returns.csv", index_col=0, parse_dates=True)
df = df.fillna(0)
df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]

# Generate rebalancing dates (first of each year)
rebal_dates = pd.date_range(df.index[0], df.index[-1], freq=f'{rebalance_months}MS')

results = []
weights_history = []
weights_dates = []

for i in range(len(rebal_dates)-1):
    train_end = rebal_dates[i] - pd.Timedelta(days=1)
    train_start = train_end - pd.DateOffset(years=window_years) + pd.Timedelta(days=1)
    test_start = rebal_dates[i]
    test_end = rebal_dates[i+1] - pd.Timedelta(days=1)
    train = df[(df.index >= train_start) & (df.index <= train_end)]
    test = df[(df.index >= test_start) & (df.index <= test_end)]
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
    weights_history.append(weights)
    weights_dates.append(test_start)
    # Apply weights to test period
    test_returns = test.dot(weights)
    results.append(test_returns)

# Concatenate out-of-sample returns
oos_returns = pd.concat(results)
oos_returns.index = pd.to_datetime(oos_returns.index)
oos_returns = oos_returns.sort_index()
cum_oos = (1 + oos_returns).cumprod()

# Plot equity curve
plt.figure(figsize=(12, 6))
plt.plot(cum_oos, label="Rolling Rebalance (Sharpe)")
plt.title("Rolling Rebalance Portfolio Equity Curve")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot weights over time
weights_history = np.array(weights_history)
plt.figure(figsize=(12, 6))
for i, col in enumerate(df.columns):
    plt.plot(weights_dates, weights_history[:, i], label=col)
plt.title("Portfolio Weights Over Time (Rolling Rebalance)")
plt.xlabel("Date")
plt.ylabel("Weight")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Stats
def print_stats(returns):
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    print(f"Rolling Rebalance Portfolio:")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2%}")
    print(f"  Volatility: {returns.std():.5f}")
    print("")

print_stats(oos_returns) 