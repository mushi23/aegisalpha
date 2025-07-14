import pandas as pd
from regime_hmm import fit_hmm_model
from regime_gmm import fit_gmm_model

def add_regime_features(df):
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['volatility_5'] = df['return'].rolling(window=5).std()
    df['momentum'] = df['close'] - df['close'].shift(5)
    df = df.dropna(subset=['return', 'volatility_5', 'momentum'])
    return df

def main():
    # Load the main DataFrame (must have 'pair', 'datetime', 'close', etc.)
    df = pd.read_csv("all_currencies_with_indicators.csv", parse_dates=["datetime"])
    hmm_regimes = []
    gmm_regimes = []
    for pair, group in df.groupby("pair"):
        print(f"\n=== Processing {pair} ===")
        group = add_regime_features(group)
        group = group.head(5000)  # Limit to first 5000 rows for speed/testing
        # HMM
        try:
            print(f"Fitting HMM for {pair} ...")
            hmm_df, _ = fit_hmm_model(group, n_components=2)
            hmm_df = hmm_df[["datetime", "regime_hmm", "bull_prob_hmm"]].copy()
            hmm_df["pair"] = pair
            hmm_regimes.append(hmm_df)
            print(f"Done HMM for {pair}.")
        except Exception as e:
            print(f"[WARN] HMM failed for {pair}: {e}")
        # GMM
        try:
            print(f"Fitting GMM for {pair} ...")
            gmm_df, _ = fit_gmm_model(group, n_components=2)
            gmm_df = gmm_df[["datetime", "regime_gmm", "bull_prob_gmm"]].copy()
            gmm_df["pair"] = pair
            gmm_regimes.append(gmm_df)
            print(f"Done GMM for {pair}.")
        except Exception as e:
            print(f"[WARN] GMM failed for {pair}: {e}")
    # Concatenate and save
    hmm_regimes_df = pd.concat(hmm_regimes, ignore_index=True)
    gmm_regimes_df = pd.concat(gmm_regimes, ignore_index=True)
    hmm_regimes_df.to_csv("hmm_regimes.csv", index=False)
    gmm_regimes_df.to_csv("gmm_regimes.csv", index=False)
    print("✅ Regime CSVs generated for all pairs.")

if __name__ == "__main__":
    main() 