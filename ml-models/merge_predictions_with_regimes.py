import pandas as pd
import argparse
import os

def load_and_merge_regimes(main_csv, hmm_csv, gmm_csv, output_csv):
    # Load main dataset (e.g., all_currencies_with_indicators.csv)
    print(f"Loading main data from: {main_csv}")
    df = pd.read_csv(main_csv, parse_dates=["datetime"])

    # Load HMM regimes
    print(f"Loading HMM regimes from: {hmm_csv}")
    hmm = pd.read_csv(hmm_csv, parse_dates=["datetime"])
    df = df.merge(hmm, on=["datetime", "pair"], how="left")

    # Load GMM regimes
    print(f"Loading GMM regimes from: {gmm_csv}")
    gmm = pd.read_csv(gmm_csv, parse_dates=["datetime"])
    df = df.merge(gmm, on=["datetime", "pair"], how="left")

    # Save merged result
    print(f"Saving merged dataset to: {output_csv}")
    df.to_csv(output_csv, index=False)
    print("✅ Merged regime data written.")

    # Print summary statistics
    print(f"\n📊 Summary:")
    print(f"Total rows: {len(df)}")
    print(f"Rows with HMM regimes: {df['regime_hmm'].notna().sum()}")
    print(f"Rows with GMM regimes: {df['regime_gmm'].notna().sum()}")
    print(f"Unique pairs: {df['pair'].nunique()}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all regime features into full indicator dataset.")
    parser.add_argument("--main_csv", default="all_currencies_with_indicators.csv", help="Main indicators CSV")
    parser.add_argument("--hmm_csv", default="hmm_regimes.csv", help="HMM regimes CSV")
    parser.add_argument("--gmm_csv", default="gmm_regimes.csv", help="GMM regimes CSV")
    parser.add_argument("--output", default="merged_predictions_with_regimes.csv", help="Output CSV")

    args = parser.parse_args()

    load_and_merge_regimes(
        main_csv=args.main_csv,
        hmm_csv=args.hmm_csv,
        gmm_csv=args.gmm_csv,
        output_csv=args.output
    ) 