import argparse
import pandas as pd
import time
import logging
import yaml
from lstm_predictor import get_lstm_predictions
from xgb_predictor import get_xgb_predictions
from regime_hmm import fit_hmm_model
from pipeline.merge_with_regimes import merge_with_regimes
from pipeline.corrective_filter import apply_corrective_filter
from pipeline.ensemble_signal import create_ensemble_signal

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_full_pipeline(pair="eurusd", timeframe="1min", config=None):
    params = config or {}
    start_time = time.time()
    logging.info(f"Step 1: Loading data for {pair} {timeframe}")
    raw_df = pd.read_csv(params.get('input_csv', f"data/{pair}_{timeframe}_indicators.csv"))
    features = params.get('features', ['close', 'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal'])

    logging.info("Step 2: LSTM/XGB predictions")
    t0 = time.time()
    lstm_preds, _, scaler = get_lstm_predictions(params.get('lstm_model', "lstm_model.h5"), raw_df, features)
    xgb_preds, _ = get_xgb_predictions(params.get('xgb_model', "xgboost_model.pkl"), raw_df)
    raw_df = raw_df.iloc[-len(lstm_preds):].copy()
    raw_df['lstm_pred'] = lstm_preds.flatten()
    raw_df['xgb_pred'] = xgb_preds
    logging.info(f"LSTM/XGB predictions done in {time.time() - t0:.2f}s")

    logging.info("Step 3: Regime detection and merging")
    t0 = time.time()
    fit_hmm_model(raw_df)  # generates hmm_regimes.csv
    merge_with_regimes(
        base_csv=params.get('input_csv', f"data/{pair}_{timeframe}_indicators.csv"),
        hmm_csv=params.get('hmm_csv', "hmm_regimes.csv"),
        gmm_csv=params.get('gmm_csv', "gmm_regimes.csv"),
        pred_dir=params.get('pred_dir', "models/predictions"),
        output_csv=params.get('merged_csv', "merged_predictions_with_regimes.csv")
    )
    logging.info(f"Regime merge done in {time.time() - t0:.2f}s")

    logging.info("Step 4: Corrective filter and ensemble signal")
    t0 = time.time()
    df = pd.read_csv(params.get('merged_csv', "merged_predictions_with_regimes.csv"))
    # Optionally apply corrective filter here if needed
    # df = apply_corrective_filter('corrective_ai_model.pkl', X_new, df, threshold=0.5)
    df = create_ensemble_signal(
        df,
        weights=tuple(params.get('ensemble_weights', (0.4, 0.4, 0.2))),
        threshold=params.get('ensemble_threshold', 0.5),
        corrective_col=params.get('corrective_col', None)
    )
    output_path = params.get('output_csv', f"output/final_{pair}_{timeframe}.csv")
    df.to_csv(output_path, index=False)
    logging.info(f"Ensemble signal and output saved in {time.time() - t0:.2f}s")

    logging.info(f"✅ Pipeline complete. Total time: {time.time() - start_time:.2f}s. Output: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="eurusd")
    parser.add_argument("--timeframe", default="1min")
    parser.add_argument("--config", type=str, default=None, help="YAML config file for pipeline parameters")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else None
    run_full_pipeline(pair=args.pair, timeframe=args.timeframe, config=config) 