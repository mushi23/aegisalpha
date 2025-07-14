#!/usr/bin/env python3
"""
Backtest the new models trained on 5-bar forward return targets.
This script will test both regression and classification models.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_models_and_data():
    """Load trained models and test data."""
    print("Loading models and data...")
    
    # Load models
    models = {}
    
    # Regression models
    try:
        models['xgb_regression'] = joblib.load('models_future_return_5_regression/xgboost_future_return_5_regression.pkl')
        models['rf_regression'] = joblib.load('models_future_return_5_regression/randomforest_future_return_5_regression.pkl')
        print("✅ Loaded regression models")
    except:
        print("❌ Could not load regression models")
    
    # Classification models
    try:
        models['xgb_binary'] = joblib.load('models_target_binary_classification/xgboost_target_binary_classification.pkl')
        models['rf_binary'] = joblib.load('models_target_binary_classification/randomforest_target_binary_classification.pkl')
        print("✅ Loaded binary classification models")
    except:
        print("❌ Could not load binary classification models")
    
    # Load scaler
    try:
        scaler = joblib.load('feature_scaler.pkl')
        print("✅ Loaded feature scaler")
    except:
        print("❌ Could not load feature scaler")
        scaler = None
    
    # Load test data
    df = pd.read_csv("all_currencies_with_indicators_updated.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Feature columns
    feature_cols = [
        'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_mid', 'support', 'resistance'
    ]
    
    return models, scaler, df, feature_cols

def generate_signals(df, models, scaler, feature_cols, threshold=0.001):
    """Generate trading signals using the trained models."""
    print("Generating trading signals...")
    
    # Prepare features
    X = df[feature_cols].values
    X_scaled = scaler.transform(X) if scaler else X
    
    signals = pd.DataFrame(index=df.index)
    signals['datetime'] = df['datetime']
    signals['pair'] = df['pair']
    signals['close'] = df['close']
    signals['future_return_5'] = df['future_return_5']
    
    # Regression model signals
    if 'xgb_regression' in models:
        xgb_pred = models['xgb_regression'].predict(X_scaled)
        signals['xgb_regression_signal'] = np.where(xgb_pred > threshold, 1, 
                                                   np.where(xgb_pred < -threshold, -1, 0))
        signals['xgb_regression_pred'] = xgb_pred
    
    if 'rf_regression' in models:
        rf_pred = models['rf_regression'].predict(X_scaled)
        signals['rf_regression_signal'] = np.where(rf_pred > threshold, 1, 
                                                  np.where(rf_pred < -threshold, -1, 0))
        signals['rf_regression_pred'] = rf_pred
    
    # Binary classification signals
    if 'xgb_binary' in models:
        xgb_bin_pred = models['xgb_binary'].predict(X_scaled)
        signals['xgb_binary_signal'] = np.where(xgb_bin_pred == 1, 1, 0)
        signals['xgb_binary_prob'] = models['xgb_binary'].predict_proba(X_scaled)[:, 1]
    
    if 'rf_binary' in models:
        rf_bin_pred = models['rf_binary'].predict(X_scaled)
        signals['rf_binary_signal'] = np.where(rf_bin_pred == 1, 1, 0)
        signals['rf_binary_prob'] = models['rf_binary'].predict_proba(X_scaled)[:, 1]
    
    return signals

def calculate_returns(signals, slippage=0.002, amplification=1.0):
    """Calculate strategy returns with slippage and amplification."""
    print("Calculating strategy returns...")
    
    results = {}
    
    # Get signal columns
    signal_cols = [col for col in signals.columns if 'signal' in col]
    
    for signal_col in signal_cols:
        strategy_name = signal_col.replace('_signal', '')
        
        # Calculate returns
        signal = signals[signal_col].values
        future_return = signals['future_return_5'].values
        
        # Apply amplification and slippage
        strategy_return = signal * future_return * amplification
        strategy_return = np.where(signal != 0, strategy_return - slippage, 0)
        
        # Clip returns to prevent overflow
        strategy_return = np.clip(strategy_return, -0.5, 0.5)
        
        # Cumulative returns
        cumulative_return = np.cumprod(1 + strategy_return) - 1
        
        # Store results
        results[strategy_name] = {
            'signal': signal,
            'strategy_return': strategy_return,
            'cumulative_return': cumulative_return,
            'total_return': cumulative_return[-1] if len(cumulative_return) > 0 else 0,
            'sharpe_ratio': np.mean(strategy_return) / np.std(strategy_return) if np.std(strategy_return) > 0 else 0,
            'win_rate': np.mean(strategy_return > 0) if np.sum(signal != 0) > 0 else 0,
            'num_trades': np.sum(signal != 0)
        }
    
    return results

def analyze_by_currency_pair(signals, results):
    """Analyze results by currency pair."""
    print("Analyzing results by currency pair...")
    
    pair_results = {}
    
    for pair in signals['pair'].unique():
        pair_mask = signals['pair'] == pair
        pair_signals = signals[pair_mask]
        
        pair_results[pair] = {}
        
        for strategy_name, strategy_data in results.items():
            pair_strategy_return = strategy_data['strategy_return'][pair_mask]
            pair_signal = strategy_data['signal'][pair_mask]
            
            pair_results[pair][strategy_name] = {
                'total_return': np.sum(pair_strategy_return),
                'sharpe_ratio': np.mean(pair_strategy_return) / np.std(pair_strategy_return) if np.std(pair_strategy_return) > 0 else 0,
                'win_rate': np.mean(pair_strategy_return > 0) if np.sum(pair_signal != 0) > 0 else 0,
                'num_trades': np.sum(pair_signal != 0)
            }
    
    return pair_results

def plot_results(signals, results):
    """Plot backtesting results."""
    print("Creating plots...")
    
    # Plot cumulative returns
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Cumulative returns
    plt.subplot(2, 2, 1)
    for strategy_name, strategy_data in results.items():
        plt.plot(signals['datetime'], strategy_data['cumulative_return'], 
                label=f'{strategy_name} ({strategy_data["total_return"]:.3f})')
    plt.title('Cumulative Strategy Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Return distribution
    plt.subplot(2, 2, 2)
    for strategy_name, strategy_data in results.items():
        returns = strategy_data['strategy_return']
        returns = returns[returns != 0]  # Only non-zero returns
        if len(returns) > 0:
            plt.hist(returns, alpha=0.7, label=f'{strategy_name}', bins=50)
    plt.title('Strategy Return Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Monthly returns heatmap
    plt.subplot(2, 2, 3)
    monthly_returns = {}
    for strategy_name, strategy_data in results.items():
        signals_with_returns = signals.copy()
        signals_with_returns['strategy_return'] = strategy_data['strategy_return']
        monthly_ret = signals_with_returns.groupby([signals_with_returns['datetime'].dt.year, 
                                                   signals_with_returns['datetime'].dt.month])['strategy_return'].sum()
        monthly_returns[strategy_name] = monthly_ret
    
    if monthly_returns:
        monthly_df = pd.DataFrame(monthly_returns)
        sns.heatmap(monthly_df.T, annot=True, fmt='.3f', cmap='RdYlGn', center=0)
        plt.title('Monthly Returns Heatmap')
    
    # Subplot 4: Performance metrics
    plt.subplot(2, 2, 4)
    metrics = ['total_return', 'sharpe_ratio', 'win_rate']
    strategy_names = list(results.keys())
    
    x = np.arange(len(strategy_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in strategy_names]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Strategy')
    plt.ylabel('Value')
    plt.title('Performance Metrics')
    plt.xticks(x + width, strategy_names, rotation=45)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_results_new_targets.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main backtesting function."""
    print("🚀 Starting backtest of new target models...")
    
    # Load models and data
    models, scaler, df, feature_cols = load_models_and_data()
    
    if not models:
        print("❌ No models loaded. Exiting.")
        return
    
    # Test different parameters
    print("\nTesting different parameters...")
    
    # Test different thresholds and amplifications (using more reasonable values)
    test_configs = [
        {'threshold': 0.001, 'amplification': 1.0, 'slippage': 0.002},
        {'threshold': 0.002, 'amplification': 1.0, 'slippage': 0.002},
        {'threshold': 0.001, 'amplification': 2.0, 'slippage': 0.002},
        {'threshold': 0.001, 'amplification': 3.0, 'slippage': 0.002},
    ]
    
    best_config = None
    best_return = -np.inf
    
    for config in test_configs:
        print(f"\nTesting config: {config}")
        # Generate signals with current threshold
        signals = generate_signals(df, models, scaler, feature_cols, threshold=config['threshold'])
        # Calculate returns with current amplification and slippage
        results = calculate_returns(signals, amplification=config['amplification'], slippage=config['slippage'])
        
        # Find best strategy
        for strategy_name, strategy_data in results.items():
            if strategy_data['total_return'] > best_return:
                best_return = strategy_data['total_return']
                best_config = config.copy()
                best_config['strategy'] = strategy_name
    
    print(f"\n🏆 Best configuration: {best_config}")
    print(f"Best total return: {best_return:.3f}")
    
    # Run final analysis with best config
    print(f"\nRunning final analysis with best config...")
    # Generate signals with best threshold
    signals = generate_signals(df, models, scaler, feature_cols, threshold=best_config['threshold'])
    # Calculate returns with best amplification and slippage
    final_results = calculate_returns(signals, amplification=best_config['amplification'], slippage=best_config['slippage'])
    
    # Analyze by currency pair
    pair_results = analyze_by_currency_pair(signals, final_results)
    
    # Print results
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n🔧 Configuration: {best_config}")
    
    print(f"\n📈 Overall Results:")
    for strategy_name, strategy_data in final_results.items():
        print(f"  {strategy_name}:")
        print(f"    Total Return: {strategy_data['total_return']:.3f}")
        print(f"    Sharpe Ratio: {strategy_data['sharpe_ratio']:.3f}")
        print(f"    Win Rate: {strategy_data['win_rate']:.3f}")
        print(f"    Number of Trades: {strategy_data['num_trades']}")
    
    print(f"\n🌍 Results by Currency Pair:")
    for pair, pair_data in pair_results.items():
        print(f"\n  {pair}:")
        for strategy_name, metrics in pair_data.items():
            print(f"    {strategy_name}: Return={metrics['total_return']:.3f}, "
                  f"Sharpe={metrics['sharpe_ratio']:.3f}, "
                  f"Win Rate={metrics['win_rate']:.3f}, "
                  f"Trades={metrics['num_trades']}")
    
    # Plot results
    plot_results(signals, final_results)
    
    # Save results
    results_df = pd.DataFrame({
        strategy_name: {
            'total_return': data['total_return'],
            'sharpe_ratio': data['sharpe_ratio'],
            'win_rate': data['win_rate'],
            'num_trades': data['num_trades']
        } for strategy_name, data in final_results.items()
    }).T
    
    results_df.to_csv('backtest_results_new_targets.csv')
    print(f"\n✅ Results saved to backtest_results_new_targets.csv")
    print(f"✅ Plots saved to backtest_results_new_targets.png")

if __name__ == "__main__":
    main() 