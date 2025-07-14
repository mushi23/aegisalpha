import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_backtest_metrics(df, price_col='close', signal_col='signal_raw', filtered_col='signal_filtered'):
    df = df.copy()
    df['return'] = df[price_col].pct_change().shift(-1)
    df['strategy_returns_raw'] = df[signal_col] * df['return']
    df['strategy_returns_filtered'] = df[filtered_col] * df['return']
    # Subtract cost per trade (only when a trade is taken)
    cost_per_trade = 0.002 + 0.005
    df['strategy_returns_raw'] -= df[signal_col] * cost_per_trade
    df['strategy_returns_filtered'] -= df[filtered_col] * cost_per_trade
    df['raw_cum_return'] = (1 + df['strategy_returns_raw']).cumprod()
    df['filtered_cum_return'] = (1 + df['strategy_returns_filtered']).cumprod()
    sharpe_raw = df['strategy_returns_raw'].mean() / df['strategy_returns_raw'].std() * np.sqrt(252)
    sharpe_filtered = df['strategy_returns_filtered'].mean() / df['strategy_returns_filtered'].std() * np.sqrt(252)
    raw_drawdown = (df['raw_cum_return'] / df['raw_cum_return'].cummax() - 1).min()
    filtered_drawdown = (df['filtered_cum_return'] / df['filtered_cum_return'].cummax() - 1).min()
    metrics = {
        'raw_final_return': df['raw_cum_return'].iloc[-1] - 1,
        'filtered_final_return': df['filtered_cum_return'].iloc[-1] - 1,
        'sharpe_raw': sharpe_raw,
        'sharpe_filtered': sharpe_filtered,
        'raw_drawdown': raw_drawdown,
        'filtered_drawdown': filtered_drawdown
    }
    return metrics, df

def run_backtest(input_csv):
    df = pd.read_csv(input_csv, parse_dates=['datetime'])
    if 'signal_raw' not in df.columns:
        df['signal_raw'] = (df['xgb_pred'] > 0.5).astype(int)
    if 'signal_filtered' not in df.columns and 'corrected_signal' in df.columns:
        df['signal_filtered'] = df['corrected_signal']
    metrics, df = compute_backtest_metrics(df)
    print("\nBacktest Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    df.set_index('datetime')[['raw_cum_return', 'filtered_cum_return']].plot(figsize=(12, 5))
    plt.title("Cumulative Returns: Raw vs Filtered")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to merged CSV with predictions and regimes')
    args = parser.parse_args()
    run_backtest(args.input) 