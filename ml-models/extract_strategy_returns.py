#!/usr/bin/env python3
"""
Extract Strategy Returns for Portfolio Optimization
Extracts per-currency strategy returns using tuned model signals and true returns.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')

class StrategyReturnExtractor:
    def __init__(self, transaction_cost=0.0, slippage=0.0):
        """
        Initialize strategy return extractor
        
        Args:
            transaction_cost: Fixed transaction cost as decimal
            slippage: Slippage per trade as decimal (e.g., 0.002 = 0.2%)
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
    def load_tuned_model_and_data(self, data_path):
        """Load tuned model and prepare data for all currency pairs"""
        print("🔄 Loading tuned model and data...")
        
        # Load data
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # Create label if needed
        if 'label' not in data.columns:
            if 'return' not in data.columns and 'close' in data.columns:
                data['return'] = data['close'].pct_change()
            cost_per_trade = 0.002 + 0.005
            data['label'] = ((data['return'] - cost_per_trade) > 0).astype(int)
        
        # Load feature list
        try:
            with open('feature_list_available.txt', 'r') as f:
                features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            features = [f for f in features if f in data.columns]
            print(f"✅ Loaded {len(features)} features")
        except FileNotFoundError:
            # Fallback to common technical features
            features = ['sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_mid']
            features = [f for f in features if f in data.columns]
            print(f"⚠️ Using fallback features: {len(features)} features")
        
        self.features = features
        return data
    
    def generate_signals_per_pair(self, data, threshold=0.001, use_regression=True, transaction_cost=0.0, slippage=0.002, percentile_threshold=None):
        """Generate trading signals for each currency pair using per-pair models (regression or classification)"""
        print("🔄 Generating signals per currency pair...")
        all_signals = []
        all_data = []
        
        for pair in data['pair'].unique():
            print(f"  Processing {pair}...")
            
            # Get data for this pair
            pair_data = data[data['pair'] == pair].copy()
            pair_data = pair_data.sort_values('datetime').reset_index(drop=True)
            
            # Prepare features and drop NaN
            with open('feature_list_enhanced.txt', 'r') as f:
                feature_list = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            missing_features = [f for f in feature_list if f not in pair_data.columns]
            if missing_features:
                raise ValueError(f"Missing features for {pair}: {missing_features}")
            
            sequence_length = 10  # Must match training
            X_sequences = []
            row_indices = []
            for i in range(sequence_length, len(pair_data)):
                window = pair_data.iloc[i-sequence_length:i][feature_list]
                if window.isnull().values.any():
                    continue  # skip if any NaNs in the window
                X_sequences.append(window.values.flatten())
                row_indices.append(i)
            
            if not X_sequences:
                print(f"    ⚠️ No valid data for {pair}, skipping")
                continue
            
            X_sequences = np.array(X_sequences)
            print(f"[DEBUG] Passing {X_sequences.shape[1]} features to model for {pair} (flattened window)")
            print(f"[DEBUG] X_sequences shape: {X_sequences.shape}")
            
            # Get the corresponding rows for predictions
            pair_data_clean = pair_data.iloc[row_indices].copy()
            
            if use_regression:
                # Load per-pair regressor and scaler
                model_path = f"models/xgboost/xgb_regressor_{pair}.pkl"
                scaler_path = f"models/scalers/xgb_regressor_scaler_{pair}.pkl"
                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    print(f"    ❌ Regressor or scaler not found for {pair}: {model_path}")
                    continue
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                X_scaled = scaler.transform(X_sequences)
                predicted_return = model.predict(X_scaled)
                execution_cost = transaction_cost + slippage
                pair_data_clean['predicted_return'] = predicted_return
                # Print predicted return stats
                print("🔍 Predicted Return Stats for {}:".format(pair))
                print("Mean:", np.mean(predicted_return))
                print("Median:", np.median(predicted_return))
                print("Max:", np.max(predicted_return))
                print("Min:", np.min(predicted_return))
                # Soft signal weighting always
                # Enable both long and short positions
                soft_signal = np.where(
                    predicted_return > threshold, predicted_return,
                    np.where(predicted_return < -threshold, predicted_return, 0)
                )
                soft_signal = soft_signal * 20
                if percentile_threshold is not None:
                    threshold_val = np.percentile(predicted_return, percentile_threshold)
                    print(f"Using percentile-based threshold: {percentile_threshold}th percentile = {threshold_val}")
                    soft_signal = np.where(np.abs(predicted_return) >= threshold_val, soft_signal, 0)
                else:
                    print(f"Using fixed threshold: {threshold}")
                    # Already handled above
                pair_data_clean['signal'] = soft_signal
                pair_data_clean['signal_proba'] = predicted_return  # For compatibility
            else:
                # Load per-pair classifier and scaler
                model_path = f"models/xgboost/xgb_model_{pair}.pkl"
                if not os.path.exists(model_path):
                    print(f"    ❌ Model not found for {pair}: {model_path}")
                    continue
                model = joblib.load(model_path)
                y_proba = model.predict_proba(X_sequences)[:, 1]
                signals = (y_proba > threshold).astype(int)
                pair_data_clean['signal'] = signals
                pair_data_clean['signal_proba'] = y_proba
            
            all_signals.append(pair_data_clean)
            all_data.append(pair_data_clean)
        
        # Combine all results
        if all_signals:
            combined_signals = pd.concat(all_signals, ignore_index=True)
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"✅ Generated signals for {len(combined_signals)} samples across {len(data['pair'].unique())} pairs")
            return combined_signals, combined_data
        else:
            print("❌ No valid signals generated")
            return None, None
    
    def calculate_strategy_returns(self, data):
        """Calculate strategy returns: signal * true_return"""
        print("🔄 Calculating strategy returns...")

        # Read filter values from environment variables (with defaults)
        rsi_min = float(os.environ.get('RSI_MIN', 0))
        rsi_max = float(os.environ.get('RSI_MAX', 100))
        vol_max = float(os.environ.get('VOL_MAX', 0.03))
        print(f"[Filter Settings] RSI_MIN={rsi_min}, RSI_MAX={rsi_max}, VOL_MAX={vol_max}")

        # Ensure we have return column
        if 'return' not in data.columns and 'close' in data.columns:
            data['return'] = data['close'].pct_change()

        # After signal generation, add expected return > 1.5 * cost filter
        cost = self.transaction_cost + self.slippage
        if 'signal_proba' in data.columns and 'return' in data.columns:
            data['expected_return'] = data['signal_proba'] * data['return']
            # data.loc[data['expected_return'] < 1.5 * cost, 'signal'] = 0
        # Add RSI, volatility filters if columns exist
        if 'rsi' in data.columns:
            data.loc[(data['rsi'] < rsi_min) | (data['rsi'] > rsi_max), 'signal'] = 0
        # if 'regime_hmm' in data.columns:
        #     # Example: only trade in bull regime (regime_hmm == 1)
        #     data.loc[data['regime_hmm'] != 1, 'signal'] = 0
        # if 'regime' in data.columns:
        #     data.loc[data['regime'] != 'bull', 'signal'] = 0
        if 'volatility_5' in data.columns:
            data.loc[data['volatility_5'] > vol_max, 'signal'] = 0

        # Print total number of trades
        print(f"Total trades: {(data['signal'] != 0).sum()}")

        # Calculate strategy returns
        data['strategy_return'] = data['signal'] * data['return']

        # Apply transaction costs and slippage on entry/exit
        if self.transaction_cost > 0 or self.slippage > 0:
            trade_change = data['signal'].diff().abs()  # 1 on entry/exit
            total_cost = trade_change * (self.transaction_cost + self.slippage)
            data['strategy_return'] -= total_cost

        return data
    
    def create_returns_matrix(self, data):
        """Create a matrix of strategy returns with datetime index and pair columns"""
        print("🔄 Creating returns matrix...")
        
        # Pivot to get returns matrix
        returns_matrix = data.pivot_table(
            index='datetime',
            columns='pair',
            values='strategy_return',
            aggfunc='first'
        )
        
        # Sort by datetime
        returns_matrix = returns_matrix.sort_index()
        
        # Fill NaN with 0 (no trade)
        returns_matrix = returns_matrix.fillna(0)
        
        print(f"✅ Returns matrix created: {returns_matrix.shape}")
        print(f"   Date range: {returns_matrix.index.min()} to {returns_matrix.index.max()}")
        print(f"   Currency pairs: {list(returns_matrix.columns)}")
        
        return returns_matrix
    
    def save_returns_matrix(self, returns_matrix, output_path):
        """Save returns matrix to CSV"""
        returns_matrix.to_csv(output_path)
        print(f"✅ Strategy returns saved to: {output_path}")
        
        # Print summary statistics
        print("\n📊 Strategy Returns Summary:")
        print(f"Total observations: {len(returns_matrix)}")
        print(f"Currency pairs: {len(returns_matrix.columns)}")
        print(f"Average return per pair:")
        for pair in returns_matrix.columns:
            avg_return = returns_matrix[pair].mean()
            print(f"  {pair}: {avg_return:.6f}")
    
    def run_extraction(self, data_path, output_path, threshold=0.001, percentile_threshold=None):
        """Run complete extraction pipeline"""
        print("🚀 Starting strategy returns extraction...")
        
        # Load data
        data = self.load_tuned_model_and_data(data_path)
        if data is None:
            return None
        
        # Generate signals
        signals_data, full_data = self.generate_signals_per_pair(data, threshold, percentile_threshold=percentile_threshold)
        if signals_data is None:
            return None
        
        # Calculate strategy returns
        signals_data = self.calculate_strategy_returns(signals_data)

        # Save signals_data with predictions and features for per-trade backtesting
        signals_data.to_csv("signals_with_predictions.csv", index=False)
        print("✅ Saved signals with predictions to signals_with_predictions.csv")
        
        # Create returns matrix
        returns_matrix = self.create_returns_matrix(signals_data)
        
        # Save results
        self.save_returns_matrix(returns_matrix, output_path)
        
        return returns_matrix

def main():
    parser = argparse.ArgumentParser(description="Extract strategy returns for portfolio optimization")
    parser.add_argument("--data", type=str, default="enhanced_regime_features.csv",
                       help="Path to enhanced dataset")
    parser.add_argument("--output", type=str, default="strategy_returns.csv",
                       help="Output path for strategy returns matrix")
    parser.add_argument("--threshold", type=float, default=0.001,
                       help="Signal threshold for model predictions (fixed threshold, regression mode)")
    parser.add_argument("--percentile-threshold", type=float, default=None,
                       help="If set, use this percentile of predicted returns as threshold (e.g., 95 for top 5%)")
    parser.add_argument("--transaction_cost", type=float, default=0.0,
                       help="Transaction cost as decimal")
    parser.add_argument("--slippage", type=float, default=0.002,
                       help="Slippage per trade as decimal (e.g., 0.002 = 0.2%)")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = StrategyReturnExtractor(transaction_cost=args.transaction_cost, slippage=args.slippage)
    
    # Run extraction
    returns_matrix = extractor.run_extraction(
        data_path=args.data,
        output_path=args.output,
        threshold=args.threshold,
        percentile_threshold=args.percentile_threshold
    )
    
    if returns_matrix is not None:
        print("\n🎉 Strategy returns extraction completed successfully!")
        print(f"📁 Output file: {args.output}")
        print(f"   Transaction cost: {args.transaction_cost}")
        print(f"   Slippage: {args.slippage}")
    else:
        print("\n❌ Strategy returns extraction failed!")

if __name__ == "__main__":
    main() 