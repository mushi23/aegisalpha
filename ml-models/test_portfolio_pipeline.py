#!/usr/bin/env python3
"""
Quick Test of Portfolio Pipeline
Tests the pipeline with a small sample to ensure everything works.
"""

import pandas as pd
import numpy as np
import os
import subprocess
import sys

CRITICAL_FEATURES = ['sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal']

def test_data_loading():
    """Test if we can load the required data"""
    print("🔄 Testing data loading...")
    
    try:
        # Load enhanced dataset
        df = pd.read_csv("enhanced_regime_features.csv")
        print(f"✅ Enhanced dataset loaded: {df.shape}")
        
        # Check required columns
        required_cols = ['datetime', 'pair', 'return', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        
        # Check currency pairs
        pairs = df['pair'].unique()
        print(f"✅ Currency pairs found: {pairs}")
        
        # Sample data for testing
        sample_df = df.head(1000)  # First 1000 rows for testing
        sample_df.to_csv("test_sample.csv", index=False)
        print("✅ Created test sample: test_sample.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_loading():
    """Test if we can load the tuned model"""
    print("🔄 Testing model loading...")
    
    try:
        import joblib
        model = joblib.load('lgbm_best_model.pkl')
        print("✅ Tuned model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_feature_list():
    """Test if feature list is valid"""
    print("🔄 Testing feature list...")
    
    try:
        with open('feature_list_full_technical.txt', 'r') as f:
            features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"✅ Feature list loaded: {len(features)} features")
        
        # Check if features exist in data
        df = pd.read_csv("enhanced_regime_features.csv")
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        print(f"✅ Available features: {len(available_features)}")
        missing_critical = [f for f in CRITICAL_FEATURES if f not in available_features]
        if missing_critical:
            print(f"❗ CRITICAL missing features: {missing_critical}")
        
        return len(available_features) > 0
        
    except Exception as e:
        print(f"❌ Feature list test failed: {e}")
        return False

def test_extract_returns():
    """Test strategy returns extraction with sample data"""
    print("🔄 Testing strategy returns extraction...")
    
    try:
        cmd = "python extract_strategy_returns.py --data test_sample.csv --output test_returns.csv --threshold 0.5"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Strategy returns extraction successful")
            
            # Check output
            if os.path.exists("test_returns.csv"):
                returns_df = pd.read_csv("test_returns.csv", index_col=0, parse_dates=True)
                print(f"✅ Returns matrix created: {returns_df.shape}")
                return True
            else:
                print("❌ Returns file not created")
                return False
        else:
            print(f"❌ Extraction failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Extraction test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    test_files = ["test_sample.csv", "test_returns.csv"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🧹 Cleaned up: {file}")

def main():
    print("🧪 Testing Portfolio Pipeline Components")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading),
        ("Feature List", test_feature_list),
        ("Returns Extraction", test_extract_returns)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is ready to run.")
        print("\n🚀 To run the full pipeline:")
        print("python run_portfolio_pipeline.py --data enhanced_regime_features.csv --output_dir portfolio_results")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the issues above.")
    
    # Cleanup
    cleanup_test_files()

if __name__ == "__main__":
    main() 