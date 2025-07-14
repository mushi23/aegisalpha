import argparse
import pandas as pd


def create_ensemble_signal(df, weights=(0.4, 0.4, 0.2), threshold=0.5, corrective_col=None):
    w_lstm, w_xgb, w_regime = weights
    df = df.copy()
    df['ensemble_score'] = (
        w_lstm * df['lstm_pred'] +
        w_xgb * df['xgb_pred'] +
        w_regime * df['bull_prob_hmm']
    )
    # If corrective_col is provided, mask ensemble signal by corrective AI output
    if corrective_col and corrective_col in df.columns:
        mask = df[corrective_col] >= threshold
        df['ensemble_signal'] = ((df['ensemble_score'] > threshold) & mask).astype(int)
    else:
        df['ensemble_signal'] = (df['ensemble_score'] > threshold).astype(int)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV with predictions and regime features')
    parser.add_argument('--output', required=True, help='Output CSV with ensemble signal')
    parser.add_argument('--weights', nargs=3, type=float, default=[0.4, 0.4, 0.2], help='Weights for LSTM, XGB, Regime')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for ensemble signal')
    parser.add_argument('--corrective_col', type=str, default=None, help='Column for corrective AI output (optional)')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = create_ensemble_signal(df, weights=tuple(args.weights), threshold=args.threshold, corrective_col=args.corrective_col)
    df.to_csv(args.output, index=False)
    print(f"✅ Ensemble signal saved to {args.output}") 