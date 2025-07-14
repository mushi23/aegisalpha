import pandas as pd
import os
import numpy as np

def merge_with_regimes(base_csv, hmm_csv, gmm_csv, pred_dir, output_csv):
    base_df = pd.read_csv(base_csv, parse_dates=["datetime"])
    base_df = base_df.sort_values(["pair", "datetime"]).reset_index(drop=True)
    hmm_regimes = pd.read_csv(hmm_csv, parse_dates=["datetime"])
    gmm_regimes = pd.read_csv(gmm_csv, parse_dates=["datetime"])
    merged_rows = []
    pairs = base_df["pair"].unique()
    for pair in pairs:
        pair_df = base_df[base_df["pair"] == pair].copy()
        pair_df = pair_df.sort_values("datetime").reset_index(drop=True)
        # LSTM predictions
        lstm_pred_path = os.path.join(pred_dir, f"lstm_pred_{pair}.npy")
        xgb_pred_path = os.path.join(pred_dir, f"xgb_pred_{pair}.npy")
        if os.path.exists(lstm_pred_path):
            lstm_preds = np.load(lstm_pred_path)
            pair_df["lstm_pred"] = lstm_preds[-len(pair_df):]
        if os.path.exists(xgb_pred_path):
            xgb_preds = np.load(xgb_pred_path)
            pair_df["xgb_pred"] = xgb_preds[-len(pair_df):]
        # Merge regimes
        pair_df = pair_df.merge(hmm_regimes, on="datetime", how="left")
        pair_df = pair_df.merge(gmm_regimes, on="datetime", how="left")
        merged_rows.append(pair_df)
    merged_df = pd.concat(merged_rows, ignore_index=True)
    merged_df.to_csv(output_csv, index=False)
    print(f"✅ Merged DataFrame saved as {output_csv}")

# Example usage (uncomment and adapt for your pipeline):
# merge_with_regimes(
#     base_csv='all_currencies_with_indicators.csv',
#     hmm_csv='hmm_regimes.csv',
#     gmm_csv='gmm_regimes.csv',
#     pred_dir='models/predictions',
#     output_csv='merged_predictions_with_regimes.csv'
# ) 