import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_enhanced_features(df, reduce_bull_prob_dominance=True):
    """
    Create enhanced features to reduce overreliance on bull_prob_hmm
    """
    df = df.copy()
    
    print("🔧 Creating enhanced features...")
    
    # 1. Regime Duration Features
    print("  - Adding regime duration features...")
    for pair in df['pair'].unique():
        pair_mask = df['pair'] == pair
        
        # HMM regime duration
        df.loc[pair_mask, 'hmm_regime_duration'] = (
            df.loc[pair_mask, 'regime_hmm'] != df.loc[pair_mask, 'regime_hmm'].shift(1)
        ).cumsum()
        
        # GMM regime duration  
        df.loc[pair_mask, 'gmm_regime_duration'] = (
            df.loc[pair_mask, 'regime_gmm'] != df.loc[pair_mask, 'regime_gmm'].shift(1)
        ).cumsum()
    
    # 2. Regime Transition Features
    print("  - Adding regime transition features...")
    df['hmm_regime_change'] = (df['regime_hmm'] != df['regime_hmm'].shift(1)).astype(int)
    df['gmm_regime_change'] = (df['regime_gmm'] != df['regime_gmm'].shift(1)).astype(int)
    df['regime_change_agreement'] = (df['hmm_regime_change'] == df['gmm_regime_change']).astype(int)
    
    # 3. Bull Probability Differences and Ratios
    print("  - Adding bull probability features...")
    df['bull_prob_diff'] = df['bull_prob_hmm'] - df['bull_prob_gmm']
    df['bull_prob_ratio'] = df['bull_prob_hmm'] / (df['bull_prob_gmm'] + 1e-8)
    df['bull_prob_avg'] = (df['bull_prob_hmm'] + df['bull_prob_gmm']) / 2

    # 3b. Bull Probability Transforms (for model compatibility)
    df['bull_prob_hmm_capped'] = np.clip(df['bull_prob_hmm'], 0.1, 0.9)
    df['bull_prob_hmm_log'] = np.log(df['bull_prob_hmm'] + 1e-8)
    df['bull_prob_hmm_sqrt'] = np.sqrt(df['bull_prob_hmm'])
    df['bull_prob_hmm_zscore'] = (df['bull_prob_hmm'] - df['bull_prob_hmm'].mean()) / df['bull_prob_hmm'].std()
    
    # 4. Volatility-Regime Interactions (using available volatility features)
    print("  - Adding volatility-regime interactions...")
    if 'regime_hmm_vol' in df.columns:
        df['vol_regime_hmm'] = df['regime_hmm_vol'] * df['regime_hmm']
    if 'regime_gmm_vol' in df.columns:
        df['vol_regime_gmm'] = df['regime_gmm_vol'] * df['regime_gmm']
    if 'regime_hmm_vol' in df.columns:
        df['vol_bull_prob_hmm'] = df['regime_hmm_vol'] * df['bull_prob_hmm']
    if 'regime_gmm_vol' in df.columns:
        df['vol_bull_prob_gmm'] = df['regime_gmm_vol'] * df['bull_prob_gmm']
    
    # 5. Return-Regime Interactions (using return instead of momentum)
    print("  - Adding return-regime interactions...")
    if 'return' in df.columns:
        df['return_regime_hmm'] = df['return'] * df['regime_hmm']
        df['return_regime_gmm'] = df['return'] * df['regime_gmm']
        df['return_bull_prob_hmm'] = df['return'] * df['bull_prob_hmm']
        df['return_bull_prob_gmm'] = df['return'] * df['bull_prob_gmm']
    
    # 6. Regime Volatility Interactions
    print("  - Adding regime volatility interactions...")
    df['regime_vol_interaction'] = df['regime_hmm_vol'] * df['regime_gmm_vol']
    df['regime_vol_diff'] = df['regime_hmm_vol'] - df['regime_gmm_vol']
    
    # 7. Rolling Statistics for Regime Features
    print("  - Adding rolling regime statistics...")
    for pair in df['pair'].unique():
        pair_mask = df['pair'] == pair
        
        # Rolling mean of bull probabilities
        df.loc[pair_mask, 'bull_prob_hmm_ma5'] = df.loc[pair_mask, 'bull_prob_hmm'].rolling(5).mean()
        df.loc[pair_mask, 'bull_prob_gmm_ma5'] = df.loc[pair_mask, 'bull_prob_gmm'].rolling(5).mean()
        
        # Rolling std of bull probabilities
        df.loc[pair_mask, 'bull_prob_hmm_std5'] = df.loc[pair_mask, 'bull_prob_hmm'].rolling(5).std()
        df.loc[pair_mask, 'bull_prob_gmm_std5'] = df.loc[pair_mask, 'bull_prob_gmm'].rolling(5).std()
    
    # 8. Feature Normalization/Reduction
    if reduce_bull_prob_dominance:
        print("  - Reducing bull_prob_hmm dominance...")
        
        # Option 1: Cap bull_prob_hmm
        df['bull_prob_hmm_capped'] = np.clip(df['bull_prob_hmm'], 0.1, 0.9)
        
        # Option 2: Log transform
        df['bull_prob_hmm_log'] = np.log(df['bull_prob_hmm'] + 1e-8)
        
        # Option 3: Square root transform
        df['bull_prob_hmm_sqrt'] = np.sqrt(df['bull_prob_hmm'])
        
        # Option 4: Z-score normalization
        scaler = StandardScaler()
        df['bull_prob_hmm_zscore'] = scaler.fit_transform(df[['bull_prob_hmm']])
    
    # 9. Market Microstructure Features
    print("  - Adding market microstructure features...")
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['volume_price_ratio'] = df['volume'] / (df['close'] * 1000)  # Normalize volume
    
    # 10. Time-based Features
    print("  - Adding time-based features...")
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 11. Regime Strength Features
    print("  - Adding regime strength features...")
    df['regime_strength_hmm'] = np.abs(df['bull_prob_hmm'] - 0.5) * 2  # 0 to 1 scale
    df['regime_strength_gmm'] = np.abs(df['bull_prob_gmm'] - 0.5) * 2
    df['regime_strength_avg'] = (df['regime_strength_hmm'] + df['regime_strength_gmm']) / 2
    
    print(f"✅ Enhanced features created! Total features: {len(df.columns)}")
    
    return df

def select_feature_sets(df, feature_set='enhanced'):
    """
    Select different feature sets for testing
    """
    base_features = [
        'return', 'regime_hmm', 'regime_gmm', 'regime_hmm_vol', 'regime_gmm_vol', 'regime_agreement'
    ]
    
    if feature_set == 'original':
        # Original features only
        features = base_features + ['bull_prob_hmm', 'bull_prob_gmm']
        
    elif feature_set == 'no_bull_prob':
        # Remove bull_prob features entirely
        features = base_features + [
            'bull_prob_diff', 'bull_prob_avg', 'regime_strength_avg',
            'hmm_regime_duration', 'gmm_regime_duration',
            'vol_regime_hmm', 'vol_regime_gmm'
        ]
        
    elif feature_set == 'reduced_bull_prob':
        # Use transformed bull_prob features
        features = base_features + [
            'bull_prob_hmm_capped', 'bull_prob_gmm',
            'bull_prob_diff', 'bull_prob_avg', 'regime_strength_avg'
        ]
        
    elif feature_set == 'enhanced':
        # All enhanced features
        features = [col for col in df.columns if col not in [
            'datetime', 'open', 'high', 'low', 'close', 'volume', 'pair',
            'actual_return', 'label'  # Target variables
        ]]
        
    else:
        features = base_features + ['bull_prob_hmm', 'bull_prob_gmm']
    
    # Ensure all features exist in dataframe
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
    
    print(f"📊 Selected {len(available_features)} features for '{feature_set}' set")
    return available_features

def main():
    parser = argparse.ArgumentParser(description="Enhanced feature engineering for regime-based trading")
    parser.add_argument("--input", default="merged_with_regime_features.csv", help="Input CSV file")
    parser.add_argument("--output", default="enhanced_regime_features.csv", help="Output CSV file")
    parser.add_argument("--feature_set", default="enhanced", 
                       choices=['original', 'no_bull_prob', 'reduced_bull_prob', 'enhanced'],
                       help="Feature set to use")
    parser.add_argument("--reduce_dominance", action="store_true", 
                       help="Reduce bull_prob_hmm dominance")
    
    args = parser.parse_args()
    
    # Load data
    print(f"📂 Loading data from {args.input}...")
    df = pd.read_csv(args.input, parse_dates=["datetime"])
    
    # Create enhanced features
    df_enhanced = create_enhanced_features(df, args.reduce_dominance)
    
    # Select feature set
    selected_features = select_feature_sets(df_enhanced, args.feature_set)
    
    # Save enhanced dataset
    print(f"💾 Saving enhanced features to {args.output}...")
    df_enhanced.to_csv(args.output, index=False)
    
    # Save feature list
    feature_list_file = f"feature_list_{args.feature_set}.txt"
    with open(feature_list_file, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    print(f"✅ Enhanced features saved!")
    print(f"📊 Feature list saved to {feature_list_file}")
    print(f"📈 Total features available: {len(df_enhanced.columns)}")
    print(f"🎯 Selected features: {len(selected_features)}")
    
    # Feature statistics
    print(f"\n📊 Feature Statistics:")
    print(f"  Original features: {len([f for f in df.columns if f not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'pair']])}")
    print(f"  Enhanced features: {len(df_enhanced.columns)}")
    print(f"  New features added: {len(df_enhanced.columns) - len(df.columns)}")

if __name__ == "__main__":
    main() 