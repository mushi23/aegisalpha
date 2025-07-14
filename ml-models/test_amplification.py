#!/usr/bin/env python3
"""
Test script to verify signal amplification is working correctly
"""

import pandas as pd
import numpy as np
import os

def test_amplification():
    print("🧪 Testing signal amplification...")
    
    # Check if signals file exists
    if os.path.exists('signals_with_predictions.csv'):
        signals = pd.read_csv('signals_with_predictions.csv')
        
        print(f"📊 Signal Analysis:")
        print(f"  Total rows: {len(signals)}")
        print(f"  Non-zero signals: {(signals['signal'] != 0).sum()}")
        print(f"  Signal stats:")
        print(f"    Mean: {signals['signal'].mean():.8f}")
        print(f"    Max: {signals['signal'].max():.8f}")
        print(f"    Min: {signals['signal'].min():.8f}")
        print(f"    Std: {signals['signal'].std():.8f}")
        
        print(f"\n🔍 Predicted Return Analysis:")
        print(f"  Mean: {signals['predicted_return'].mean():.8f}")
        print(f"  Max: {signals['predicted_return'].max():.8f}")
        print(f"  Min: {signals['predicted_return'].min():.8f}")
        
        # Check if amplification is working
        non_zero_signals = signals[signals['signal'] != 0]
        if len(non_zero_signals) > 0:
            print(f"\n📈 Amplification Check:")
            print(f"  Sample non-zero signals:")
            for i in range(min(5, len(non_zero_signals))):
                pred = non_zero_signals.iloc[i]['predicted_return']
                signal = non_zero_signals.iloc[i]['signal']
                amplification = signal / pred if pred != 0 else 0
                print(f"    Pred: {pred:.8f} -> Signal: {signal:.8f} (Amp: {amplification:.1f}x)")
        
        # Check if 50x amplification is visible
        expected_amplification = 50
        actual_amplifications = []
        for _, row in non_zero_signals.iterrows():
            if row['predicted_return'] != 0:
                amp = abs(row['signal']) / abs(row['predicted_return'])
                actual_amplifications.append(amp)
        
        if actual_amplifications:
            avg_amp = np.mean(actual_amplifications)
            print(f"\n🎯 Amplification Analysis:")
            print(f"  Expected: {expected_amplification}x")
            print(f"  Average actual: {avg_amp:.1f}x")
            print(f"  Min actual: {min(actual_amplifications):.1f}x")
            print(f"  Max actual: {max(actual_amplifications):.1f}x")
            
            if abs(avg_amp - expected_amplification) < 5:
                print("✅ Amplification appears to be working correctly")
            else:
                print("❌ Amplification may not be working as expected")
        else:
            print("❌ No non-zero signals to analyze")
    else:
        print("❌ No signals file found. Run the strategy extraction first.")

if __name__ == "__main__":
    test_amplification() 