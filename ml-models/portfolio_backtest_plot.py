import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load strategy returns
df = pd.read_csv("strategy_returns.csv", index_col=0, parse_dates=True)
df = df.fillna(0)

# Combine all pairs equally
top_pairs = [col for col in df.columns if col in ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDJPY"]]
portfolio_returns = df[top_pairs].mean(axis=1)

# Optional: apply slippage (set to 0.0 for no slippage)
slippage = 0.0  # Change as needed, e.g., 0.0005
portfolio_returns_net = portfolio_returns - slippage * np.sign(portfolio_returns)

# Cumulative returns
portfolio_cum_returns = (1 + portfolio_returns_net).cumprod()

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(portfolio_cum_returns, label="Portfolio (Net of Slippage)")
plt.title("Cumulative Portfolio Return (Equal Weight)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print stats
sharpe = np.mean(portfolio_returns_net) / np.std(portfolio_returns_net) * np.sqrt(252) if np.std(portfolio_returns_net) > 0 else np.nan
max_drawdown = (portfolio_cum_returns / portfolio_cum_returns.cummax() - 1).min()
print(f"Portfolio Sharpe: {sharpe:.2f}")
print(f"Portfolio Max Drawdown: {max_drawdown:.2%}")

# Optional: overlay for different thresholds (if files exist)
for thresh in [0.001, 0.0, -1.0]:
    fname = f"strategy_returns_thresh{thresh}.csv"
    if os.path.exists(fname):
        df_alt = pd.read_csv(fname, index_col=0, parse_dates=True).fillna(0)
        port_alt = df_alt[top_pairs].mean(axis=1)
        port_alt_cum = (1 + port_alt).cumprod()
        plt.plot(port_alt_cum, label=f"Threshold {thresh}")
if any(os.path.exists(f"strategy_returns_thresh{t}.csv") for t in [0.001, 0.0, -1.0]):
    plt.title("Cumulative Portfolio Return (Threshold Comparison)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 