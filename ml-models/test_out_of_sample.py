#!/usr/bin/env python3
"""
Test model performance on out-of-sample data
Uses last 2 years (2011-2013) as test data to check for overfitting
"""

import pandas as pd
import numpy as np
import os

def test_out_of_sample():
    print("🧪 Testing model on out-of-sample data...")
    
    # Load the full dataset
    data = pd.read_csv('enhanced_regime_features.csv')
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Split data chronologically
    train_data = data[data['datetime'] < '2011-01-01'].copy()
    test_data = data[data['datetime'] >= '2011-01-01'].copy()
    
    print(f"📊 Data split:")
    print(f"  Training data: {len(train_data)} rows ({train_data['datetime'].min()} to {train_data['datetime'].max()})")
    print(f"  Test data: {len(test_data)} rows ({test_data['datetime'].min()} to {test_data['datetime'].max()})")
    
    # Save split datasets
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    
    print("✅ Split datasets saved to train_data.csv and test_data.csv")
    
    # Test on out-of-sample data
    print("\n🔄 Testing on out-of-sample data...")
    os.system('python extract_strategy_returns.py --data test_data.csv --output test_returns.csv --percentile-threshold 90 --transaction_cost 0.0 --slippage 0.002')
    
    # Compare results
    if os.path.exists('test_returns.csv'):
        test_results = pd.read_csv('test_returns.csv', index_col=0)
        print(f"\n📈 Out-of-Sample Results:")
        print(f"  Total observations: {len(test_results)}")
        print(f"  Average return per pair:")
        for pair in test_results.columns:
            avg_return = test_results[pair].mean()
            print(f"    {pair}: {avg_return:.6f}")
        
        # Calculate total return
        total_return = test_results.sum(axis=1).mean()
        print(f"\n  Total average return: {total_return:.6f}")
        
        if total_return > 0:
            print("✅ Model shows positive returns on out-of-sample data")
        else:
            print("❌ Model shows negative returns on out-of-sample data (likely overfitting)")
    else:
        print("❌ Test failed - no results file generated")

if __name__ == "__main__":
    test_out_of_sample() 