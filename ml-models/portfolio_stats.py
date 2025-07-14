import pandas as pd
import numpy as np

returns = pd.read_csv("strategy_returns.csv", index_col=0, parse_dates=True)
print("Per Pair Stats:")
for col in returns.columns:
    r = returns[col]
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    max_dd = (r.cumsum() - r.cumsum().cummax()).min()
    print(f"{col}: Sharpe={sharpe:.2f}, Max DD={max_dd:.2%}, Vol={r.std():.5f}")

# Portfolio (equal-weighted)
portfolio_returns = returns.mean(axis=1)
cumulative = (1 + portfolio_returns).cumprod()
sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
max_dd = (cumulative / cumulative.cummax() - 1).min()
print(f"\nPortfolio Sharpe: {sharpe:.2f}")
print(f"Portfolio Max Drawdown: {max_dd:.2%}")
print(f"Portfolio Volatility: {portfolio_returns.std():.5f}") 