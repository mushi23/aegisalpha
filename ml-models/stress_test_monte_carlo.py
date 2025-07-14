import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load rolling rebalanced returns (re-run rolling_rebalance.py if needed)
def get_rolling_returns():
    # Re-run the logic here for simplicity
    from scipy.optimize import minimize
    window_years = 5
    rebalance_months = 12
    start_year = 1993
    end_year = 2013
    df = pd.read_csv("strategy_returns.csv", index_col=0, parse_dates=True)
    df = df.fillna(0)
    df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
    rebal_dates = pd.date_range(df.index[0], df.index[-1], freq=f'{rebalance_months}MS')
    results = []
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
        test_returns = test.dot(weights)
        results.append(test_returns)
    oos_returns = pd.concat(results)
    oos_returns.index = pd.to_datetime(oos_returns.index)
    oos_returns = oos_returns.sort_index()
    return oos_returns

def print_stats(returns, label):
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    print(f"{label}:")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2%}")
    print(f"  Volatility: {returns.std():.5f}")
    print("")

# 1. Slippage/Transaction Cost Stress Test
slippages = [0.0, 0.00005, 0.0001, 0.0002]
returns = get_rolling_returns()

plt.figure(figsize=(12, 6))
for slip in slippages:
    adj_returns = returns - slip
    cum = (1 + adj_returns).cumprod()
    plt.plot(cum, label=f"Slippage={slip}")
    print_stats(adj_returns, f"Slippage={slip}")
plt.title("Equity Curve Under Different Slippage/Transaction Costs")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Monte Carlo Bootstrap
n_sim = 1000
sim_len = len(returns)
final_returns = []
sharpes = []
drawdowns = []
for _ in range(n_sim):
    sim = np.random.choice(returns, size=sim_len, replace=True)
    cum = np.cumprod(1 + sim)
    final_returns.append(cum[-1])
    sharpe = np.mean(sim) / np.std(sim) * np.sqrt(252) if np.std(sim) > 0 else np.nan
    sharpes.append(sharpe)
    max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
    drawdowns.append(max_dd)

plt.figure(figsize=(12, 6))
plt.hist(final_returns, bins=40, alpha=0.7)
plt.title("Monte Carlo: Distribution of Final Portfolio Value")
plt.xlabel("Final Cumulative Return")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(sharpes, bins=40, alpha=0.7)
plt.title("Monte Carlo: Distribution of Sharpe Ratios")
plt.xlabel("Sharpe Ratio")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(drawdowns, bins=40, alpha=0.7)
plt.title("Monte Carlo: Distribution of Max Drawdowns")
plt.xlabel("Max Drawdown")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show() 