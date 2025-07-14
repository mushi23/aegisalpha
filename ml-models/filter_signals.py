#!/usr/bin/env python3
"""
Modular Signal Filtering Script
Filters signals based on RSI, volatility, and regime.
"""

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Filter trading signals with custom rules.")
parser.add_argument('--input', type=str, required=True, help='Input signals CSV')
parser.add_argument('--output', type=str, required=True, help='Output filtered signals CSV')
parser.add_argument('--rsi', type=float, default=None, help='Minimum RSI to allow trade (e.g., 60)')
parser.add_argument('--volatility_threshold', type=float, default=None, help='Maximum volatility to allow trade (e.g., 0.03)')
parser.add_argument('--regime', type=str, default=None, help='Required regime (e.g., bull)')
parser.add_argument('--regime_hmm', type=int, default=None, help='Required regime_hmm value (e.g., 1 for bull)')

args = parser.parse_args()

df = pd.read_csv(args.input)

# Apply RSI filter
if args.rsi is not None and 'rsi' in df.columns:
    df.loc[df['rsi'] < args.rsi, 'signal'] = 0

# Apply volatility filter
if args.volatility_threshold is not None and 'volatility_5' in df.columns:
    df.loc[df['volatility_5'] > args.volatility_threshold, 'signal'] = 0

# Apply regime filter
if args.regime is not None and 'regime' in df.columns:
    df.loc[df['regime'] != args.regime, 'signal'] = 0

# Apply regime_hmm filter
if args.regime_hmm is not None and 'regime_hmm' in df.columns:
    df.loc[df['regime_hmm'] != args.regime_hmm, 'signal'] = 0

# Save filtered signals
df.to_csv(args.output, index=False)
print(f"✅ Filtered signals saved to: {args.output}") 