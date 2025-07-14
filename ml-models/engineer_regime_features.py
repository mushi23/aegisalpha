import pandas as pd
import numpy as np
import argparse

def compute_regime_volatility(df, regime_col, return_col='return', window=20):
    vol_col = f"{regime_col}_vol"
    df[vol_col] = (
        df.groupby(regime_col)[return_col]
        .transform(lambda x: x.rolling(window=window, min_periods=1).std())
    )
    return df

def engineer_regime_features(input_path="merged_predictions_with_regimes.csv", output_path="merged_with_regime_features.csv", window=20):
    df = pd.read_csv(input_path)

    # Compute return if not already there
    if 'return' not in df.columns:
        df['return'] = df['close'].pct_change()

    # Volatility per regime
    if 'regime_hmm' in df.columns:
        df = compute_regime_volatility(df, 'regime_hmm', window=window)
    if 'regime_gmm' in df.columns:
        df = compute_regime_volatility(df, 'regime_gmm', window=window)

    # Regime agreement feature
    if 'regime_hmm' in df.columns and 'regime_gmm' in df.columns:
        df['regime_agreement'] = (df['regime_hmm'] == df['regime_gmm']).astype(int)

    # Drop early NaNs
    df.dropna(inplace=True)

    # Save
    df.to_csv(output_path, index=False)
    print(f"✅ Saved with regime volatility and agreement: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='merged_predictions_with_regimes.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='merged_with_regime_features.csv', help='Output CSV file')
    parser.add_argument('--window', type=int, default=20, help='Rolling window size for volatility')
    args = parser.parse_args()
    engineer_regime_features(args.input, args.output, args.window) 