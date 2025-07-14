#!/usr/bin/env python3
"""
Test model performance on middle period (2005-2007) as additional out-of-sample validation
"""

import pandas as pd
import numpy as np
import os

def test_middle_period():
    print("🧪 Testing model on middle period (2005-2007)...")
    
    # Load the full dataset
    data = pd.read_csv('enhanced_regime_features.csv')
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Use middle period as test data
    test_data = data[(data['datetime'] >= '2005-01-01') & (data['datetime'] < '2008-01-01')].copy()
    
    print(f"📊 Middle period test data:")
    print(f"  Test data: {len(test_data)} rows ({test_data['datetime'].min()} to {test_data['datetime'].max()})")
    
    # Save test dataset
    test_data.to_csv('test_middle_period.csv', index=False)
    
    print("✅ Middle period test dataset saved to test_middle_period.csv")
    
    # Test with 1000x amplification and realistic slippage
    print("\n🔄 Testing on middle period with 1000x amplification and realistic slippage...")
    os.system('python extract_strategy_returns.py --data test_middle_period.csv --output test_middle_returns.csv --percentile-threshold 90 --transaction_cost 0.0 --slippage 0.005')
    # Test with 50x amplification and realistic slippage
    print("\n🔄 Testing on middle period with 50x amplification and realistic slippage...")
    os.system('python extract_strategy_returns.py --data test_middle_period.csv --output test_middle_returns_50x.csv --percentile-threshold 90 --transaction_cost 0.0 --slippage 0.005')
    # Test with 10x amplification and realistic slippage
    print("\n🔄 Testing on middle period with 10x amplification and realistic slippage...")
    os.system('python extract_strategy_returns.py --data test_middle_period.csv --output test_middle_returns_10x.csv --percentile-threshold 90 --transaction_cost 0.0 --slippage 0.005')
    
    # Compare results
    if os.path.exists('test_middle_returns.csv'):
        test_results = pd.read_csv('test_middle_returns.csv', index_col=0)
        print(f"\n📈 Middle Period Results (2005-2007) - 1000x amplification:")
        print(f"  Total observations: {len(test_results)}")
        print(f"  Average return per pair:")
        for pair in test_results.columns:
            avg_return = test_results[pair].mean()
            print(f"    {pair}: {avg_return:.6f}")
        
        # Calculate total return
        total_return = test_results.sum(axis=1).mean()
        print(f"\n  Total average return: {total_return:.6f}")
        
        if total_return > 0:
            print("✅ Model shows positive returns on middle period")
        else:
            print("❌ Model shows negative returns on middle period")
    else:
        print("❌ Test failed - no results file generated")
    
    # Check 50x results
    if os.path.exists('test_middle_returns_50x.csv'):
        test_results_50x = pd.read_csv('test_middle_returns_50x.csv', index_col=0)
        print(f"\n📈 Middle Period Results (2005-2007) - 50x amplification:")
        print(f"  Total observations: {len(test_results_50x)}")
        print(f"  Average return per pair:")
        for pair in test_results_50x.columns:
            avg_return = test_results_50x[pair].mean()
            print(f"    {pair}: {avg_return:.6f}")
        
        # Calculate total return
        total_return_50x = test_results_50x.sum(axis=1).mean()
        print(f"\n  Total average return: {total_return_50x:.6f}")
        
        if total_return_50x > 0:
            print("✅ Model shows positive returns on middle period with 50x")
        else:
            print("❌ Model shows negative returns on middle period with 50x")
    else:
        print("❌ 50x test failed - no results file generated")
    
    # Check 10x results
    if os.path.exists('test_middle_returns_10x.csv'):
        test_results_10x = pd.read_csv('test_middle_returns_10x.csv', index_col=0)
        print(f"\n📈 Middle Period Results (2005-2007) - 10x amplification:")
        print(f"  Total observations: {len(test_results_10x)}")
        print(f"  Average return per pair:")
        for pair in test_results_10x.columns:
            avg_return = test_results_10x[pair].mean()
            print(f"    {pair}: {avg_return:.6f}")
        
        # Calculate total return
        total_return_10x = test_results_10x.sum(axis=1).mean()
        print(f"\n  Total average return: {total_return_10x:.6f}")
        
        if total_return_10x > 0:
            print("✅ Model shows positive returns on middle period with 10x")
        else:
            print("❌ Model shows negative returns on middle period with 10x")
    else:
        print("❌ 10x test failed - no results file generated")

if __name__ == "__main__":
    test_middle_period() 