import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load strategy returns
df = pd.read_csv("strategy_returns.csv", parse_dates=True, index_col=0)
df = df.fillna(0)  # Fill any missing values with 0

# Calculate cumulative returns (additive)
cumulative_returns = df.cumsum()

# Plot cumulative returns
plt.figure(figsize=(12, 6))
for col in df.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[col], label=col)
plt.title("Cumulative Returns per Pair")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print stats
print("📈 Strategy Stats:")
for pair in df.columns:
    returns = df[pair]
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    cumret = returns.cumsum()
    max_drawdown = (cumret - cumret.cummax()).min()
    print(f"{pair}:")
    print(f"  ➤ Mean daily return: {returns.mean():.6f}")
    print(f"  ➤ Std dev: {returns.std():.6f}")
    print(f"  ➤ Sharpe Ratio: {sharpe:.2f}")
    print(f"  ➤ Max Drawdown: {max_drawdown:.6f}")
    print("")

# Optional: histogram of returns
df.plot.hist(bins=100, alpha=0.7, figsize=(10, 5), title="Daily Return Distribution")
plt.tight_layout()
plt.show() 