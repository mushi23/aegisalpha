import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Experiment settings
thresholds = [0.001, 0.002, 0.003]
slippages = [0.0005, 0.001, 0.002]
filters = [
    {'rsi_min': 0, 'rsi_max': 100, 'vol_max': 0.03, 'label': 'No Filter'},
    {'rsi_min': 0, 'rsi_max': 60, 'vol_max': 0.02, 'label': 'RSI<60, Vol<0.02'},
]
data_path = 'enhanced_regime_features.csv'
base_cmd = 'python extract_strategy_returns.py --data {data} --output {output} --transaction_cost 0.0 --slippage {slip} --threshold {thresh}'

results = []

for filt in filters:
    # Make a safe label for the filter
    safe_label = filt['label'].replace(' ', '').replace('<', '_lt_').replace(',', '_')
    for thresh in thresholds:
        for slip in slippages:
            label = f"T{thresh}_S{slip}_{safe_label}"
            output = f"strategy_returns_{label}.csv"
            # Set environment variables for filters (script should read these)
            os.environ['RSI_MIN'] = str(filt['rsi_min'])
            os.environ['RSI_MAX'] = str(filt['rsi_max'])
            os.environ['VOL_MAX'] = str(filt['vol_max'])
            # Run extraction
            cmd = base_cmd.format(data=data_path, output=output, slip=slip, thresh=thresh)
            print(f"\n=== Running: {label} ===\n{cmd}")
            os.system(cmd)
            # Load results
            if os.path.exists(output):
                df = pd.read_csv(output, index_col=0, parse_dates=True).fillna(0)
                port = df.mean(axis=1)
                cum = (1 + port).cumprod()
                # Count trades (nonzero signals)
                trade_count = (df != 0).sum().sum()
                results.append({'label': label, 'cum': cum, 'returns': port, 'trades': trade_count})

# Plot cumulative returns
plt.figure(figsize=(14, 7))
for res in results:
    plt.plot(res['cum'], label=res['label'])
plt.title('Cumulative Portfolio Return: Threshold & Slippage Experiments')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print summary stats
for res in results:
    sharpe = res['returns'].mean() / res['returns'].std() * np.sqrt(252) if res['returns'].std() > 0 else np.nan
    max_dd = (res['cum'] / res['cum'].cummax() - 1).min()
    print(f"{res['label']}: Sharpe={sharpe:.2f}, MaxDD={max_dd:.2%}, Trades={res['trades']}") 