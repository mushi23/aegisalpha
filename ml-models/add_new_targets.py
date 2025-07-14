#!/usr/bin/env python3
"""
Add new target columns to the existing all_currencies_with_indicators.csv file:
- 5-bar forward return
- 5-bar forward log return  
- Classification target (strong long, no trade, strong short)
"""

import pandas as pd
import numpy as np
import os

def add_new_targets():
    print("Loading existing dataset...")
    df = pd.read_csv("all_currencies_with_indicators.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Currency pairs: {df['pair'].unique()}")
    
    # Container for processed DataFrames
    processed_dfs = []
    
    # Process each currency pair separately
    for pair in df['pair'].unique():
        print(f"Processing {pair}...")
        pair_df = df[df['pair'] == pair].copy()
        pair_df = pair_df.sort_values('datetime').reset_index(drop=True)
        
        # Calculate 5-bar forward returns
        pair_df['future_return_5'] = (pair_df['close'].shift(-5) - pair_df['close']) / pair_df['close']
        pair_df['future_log_return_5'] = np.log(pair_df['close'].shift(-5)) - np.log(pair_df['close'])
        
        # Create classification target
        threshold = 0.002  # 0.2% threshold
        pair_df['target_class'] = 0  # no trade
        pair_df.loc[pair_df['future_return_5'] > threshold, 'target_class'] = 1  # strong long
        pair_df.loc[pair_df['future_return_5'] < -threshold, 'target_class'] = -1  # strong short
        
        # Also create a binary classification (long vs not long)
        pair_df['target_binary'] = (pair_df['future_return_5'] > threshold).astype(int)
        
        # Create more granular classification with multiple thresholds
        pair_df['target_multi'] = 0  # no trade
        pair_df.loc[pair_df['future_return_5'] > 0.005, 'target_multi'] = 2  # very strong long
        pair_df.loc[(pair_df['future_return_5'] > 0.002) & (pair_df['future_return_5'] <= 0.005), 'target_multi'] = 1  # strong long
        pair_df.loc[(pair_df['future_return_5'] < -0.002) & (pair_df['future_return_5'] >= -0.005), 'target_multi'] = -1  # strong short
        pair_df.loc[pair_df['future_return_5'] < -0.005, 'target_multi'] = -2  # very strong short
        
        processed_dfs.append(pair_df)
    
    # Combine all processed DataFrames
    df_updated = pd.concat(processed_dfs, ignore_index=True)
    
    # Remove rows where we can't calculate future returns (last 5 bars of each pair)
    df_updated = df_updated.dropna(subset=['future_return_5', 'future_log_return_5'])
    
    print(f"Updated dataset shape: {df_updated.shape}")
    print(f"Target class distribution:")
    print(df_updated['target_class'].value_counts().sort_index())
    print(f"Target multi distribution:")
    print(df_updated['target_multi'].value_counts().sort_index())
    
    # Save updated dataset
    output_file = "all_currencies_with_indicators_updated.csv"
    df_updated.to_csv(output_file, index=False)
    print(f"✅ Saved updated dataset to {output_file}")
    
    # Also save a backup of the original
    backup_file = "all_currencies_with_indicators_backup.csv"
    df.to_csv(backup_file, index=False)
    print(f"✅ Saved backup of original dataset to {backup_file}")
    
    # Show some statistics
    print("\n📊 Target Statistics:")
    for pair in df_updated['pair'].unique():
        pair_data = df_updated[df_updated['pair'] == pair]
        print(f"\n{pair}:")
        print(f"  Total samples: {len(pair_data)}")
        print(f"  Mean 5-bar return: {pair_data['future_return_5'].mean():.4f}")
        print(f"  Std 5-bar return: {pair_data['future_return_5'].std():.4f}")
        print(f"  Class distribution: {pair_data['target_class'].value_counts().sort_index().to_dict()}")
    
    return df_updated

if __name__ == "__main__":
    df_updated = add_new_targets()
    print("\n✅ Script completed successfully!") 