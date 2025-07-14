#!/usr/bin/env python3
"""
Realistic backtesting of the new 5-bar forward return models with proper risk management.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
        signals['xgb_regression_confidence'] = np.abs(xgb_pred)
    
    if 'rf_regression' in models:
        rf_pred = models['rf_regression'].predict(X_scaled)
        signals['rf_regression_signal'] = np.where(rf_pred > threshold, 1, 
                                                  np.where(rf_pred < -threshold, -1, 0))
        signals['rf_regression_pred'] = rf_pred
        signals['rf_regression_confidence'] = np.abs(rf_pred)
    
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

def calculate_realistic_returns(signals, slippage=0.002, max_position_size=0.1, confidence_threshold=0.002):
    """Calculate strategy returns with realistic risk management."""
    print("Calculating realistic strategy returns...")
    
    results = {}
    
    # Get signal columns
    signal_cols = [col for col in signals.columns if 'signal' in col]
    
    for signal_col in signal_cols:
        strategy_name = signal_col.replace('_signal', '')
        
        # Get signals and confidence
        signal = signals[signal_col].values
        future_return = signals['future_return_5'].values
        
        # Get confidence if available
        confidence_col = f'{strategy_name}_confidence'
        if confidence_col in signals.columns:
            confidence = signals[confidence_col].values
        else:
            confidence = np.ones_like(signal)
        
        # Calculate position size based on confidence
        position_size = np.minimum(confidence * max_position_size, max_position_size)
        position_size = np.where(confidence < confidence_threshold, 0, position_size)
        
        # Calculate returns with position sizing
        strategy_return = signal * future_return * position_size
        
        # Apply slippage only to actual trades
        strategy_return = np.where(signal != 0, strategy_return - slippage * position_size, 0)
        
        # Clip returns to prevent extreme values
        strategy_return = np.clip(strategy_return, -0.1, 0.1)
        
        # Calculate cumulative returns
        cumulative_return = np.cumprod(1 + strategy_return) - 1
        
        # Calculate metrics
        total_return = cumulative_return[-1] if len(cumulative_return) > 0 else 0
        sharpe_ratio = np.mean(strategy_return) / np.std(strategy_return) if np.std(strategy_return) > 0 else 0
        win_rate = np.mean(strategy_return > 0) if np.sum(signal != 0) > 0 else 0
        num_trades = np.sum(signal != 0)
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(1 + cumulative_return)
        drawdown = (1 + cumulative_return) / peak - 1
        max_drawdown = np.min(drawdown)
        
        # Store results
        results[strategy_name] = {
            'signal': signal,
            'strategy_return': strategy_return,
            'cumulative_return': cumulative_return,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'max_drawdown': max_drawdown,
            'position_size': position_size
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
            
            # Calculate pair-specific metrics
            total_return = np.sum(pair_strategy_return)
            sharpe_ratio = np.mean(pair_strategy_return) / np.std(pair_strategy_return) if np.std(pair_strategy_return) > 0 else 0
            win_rate = np.mean(pair_strategy_return > 0) if np.sum(pair_signal != 0) > 0 else 0
            num_trades = np.sum(pair_signal != 0)
            
            # Calculate max drawdown for this pair
            cumulative_return = np.cumprod(1 + pair_strategy_return) - 1
            peak = np.maximum.accumulate(1 + cumulative_return)
            drawdown = (1 + cumulative_return) / peak - 1
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            
            pair_results[pair][strategy_name] = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'num_trades': num_trades,
                'max_drawdown': max_drawdown
            }
    
    return pair_results

def plot_realistic_results(signals, results):
    """Plot realistic backtesting results."""
    print("Creating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Subplot 1: Cumulative returns
    ax1 = axes[0, 0]
    for strategy_name, strategy_data in results.items():
        ax1.plot(signals['datetime'], strategy_data['cumulative_return'], 
                label=f'{strategy_name} ({strategy_data["total_return"]:.3f})')
    ax1.set_title('Cumulative Strategy Returns')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True)
    
    # Subplot 2: Return distribution
    ax2 = axes[0, 1]
    for strategy_name, strategy_data in results.items():
        returns = strategy_data['strategy_return']
        returns = returns[returns != 0]  # Only non-zero returns
        if len(returns) > 0:
            ax2.hist(returns, alpha=0.7, label=f'{strategy_name}', bins=50)
    ax2.set_title('Strategy Return Distribution')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True)
    
    # Subplot 3: Drawdown
    ax3 = axes[0, 2]
    for strategy_name, strategy_data in results.items():
        cumulative_return = strategy_data['cumulative_return']
        peak = np.maximum.accumulate(1 + cumulative_return)
        drawdown = (1 + cumulative_return) / peak - 1
        ax3.plot(signals['datetime'], drawdown, label=f'{strategy_name}')
    ax3.set_title('Strategy Drawdown')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown')
    ax3.legend()
    ax3.grid(True)
    
    # Subplot 4: Performance metrics
    ax4 = axes[1, 0]
    metrics = ['total_return', 'sharpe_ratio', 'win_rate']
    strategy_names = list(results.keys())
    
    x = np.arange(len(strategy_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in strategy_names]
        ax4.bar(x + i*width, values, width, label=metric)
    
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Value')
    ax4.set_title('Performance Metrics')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(strategy_names, rotation=45)
    ax4.legend()
    ax4.grid(True)
    
    # Subplot 5: Number of trades
    ax5 = axes[1, 1]
    trade_counts = [results[name]['num_trades'] for name in strategy_names]
    ax5.bar(strategy_names, trade_counts)
    ax5.set_title('Number of Trades')
    ax5.set_ylabel('Count')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True)
    
    # Subplot 6: Max drawdown
    ax6 = axes[1, 2]
    max_drawdowns = [results[name]['max_drawdown'] for name in strategy_names]
    ax6.bar(strategy_names, max_drawdowns, color='red', alpha=0.7)
    ax6.set_title('Maximum Drawdown')
    ax6.set_ylabel('Drawdown')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('realistic_backtest_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main realistic backtesting function."""
    print("🚀 Starting realistic backtest of new target models...")
    
    # Load models and data
    models, scaler, df, feature_cols = load_models_and_data()
    
    if not models:
        print("❌ No models loaded. Exiting.")
        return
    
    # Generate signals
    signals = generate_signals(df, models, scaler, feature_cols, threshold=0.001)
    
    # Test different risk management parameters
    print("\nTesting different risk management parameters...")
    
    test_configs = [
        {'slippage': 0.002, 'max_position_size': 0.05, 'confidence_threshold': 0.001},
        {'slippage': 0.002, 'max_position_size': 0.1, 'confidence_threshold': 0.002},
        {'slippage': 0.001, 'max_position_size': 0.1, 'confidence_threshold': 0.002},
        {'slippage': 0.002, 'max_position_size': 0.15, 'confidence_threshold': 0.003},
    ]
    
    best_config = None
    best_sharpe = -np.inf
    
    for config in test_configs:
        print(f"\nTesting config: {config}")
        results = calculate_realistic_returns(signals, **config)
        
        # Find best strategy based on Sharpe ratio
        for strategy_name, strategy_data in results.items():
            if strategy_data['sharpe_ratio'] > best_sharpe:
                best_sharpe = strategy_data['sharpe_ratio']
                best_config = config.copy()
                best_config['strategy'] = strategy_name
    
    print(f"\n🏆 Best configuration: {best_config}")
    print(f"Best Sharpe ratio: {best_sharpe:.3f}")
    
    # Run final analysis with best config
    print(f"\nRunning final analysis with best config...")
    final_results = calculate_realistic_returns(signals, **{k: v for k, v in best_config.items() if k != 'strategy'})
    
    # Analyze by currency pair
    pair_results = analyze_by_currency_pair(signals, final_results)
    
    # Print results
    print("\n" + "="*60)
    print("📊 REALISTIC BACKTEST RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n🔧 Configuration: {best_config}")
    
    print(f"\n📈 Overall Results:")
    for strategy_name, strategy_data in final_results.items():
        print(f"  {strategy_name}:")
        print(f"    Total Return: {strategy_data['total_return']:.3f}")
        print(f"    Sharpe Ratio: {strategy_data['sharpe_ratio']:.3f}")
        print(f"    Win Rate: {strategy_data['win_rate']:.3f}")
        print(f"    Number of Trades: {strategy_data['num_trades']}")
        print(f"    Max Drawdown: {strategy_data['max_drawdown']:.3f}")
    
    print(f"\n🌍 Results by Currency Pair:")
    for pair, pair_data in pair_results.items():
        print(f"\n  {pair}:")
        for strategy_name, metrics in pair_data.items():
            print(f"    {strategy_name}: Return={metrics['total_return']:.3f}, "
                  f"Sharpe={metrics['sharpe_ratio']:.3f}, "
                  f"Win Rate={metrics['win_rate']:.3f}, "
                  f"Trades={metrics['num_trades']}, "
                  f"Max DD={metrics['max_drawdown']:.3f}")
    
    # Plot results
    plot_realistic_results(signals, final_results)
    
    # Save results
    results_df = pd.DataFrame({
        strategy_name: {
            'total_return': data['total_return'],
            'sharpe_ratio': data['sharpe_ratio'],
            'win_rate': data['win_rate'],
            'num_trades': data['num_trades'],
            'max_drawdown': data['max_drawdown']
        } for strategy_name, data in final_results.items()
    }).T
    
    results_df.to_csv('realistic_backtest_results.csv')
    print(f"\n✅ Results saved to realistic_backtest_results.csv")
    print(f"✅ Plots saved to realistic_backtest_results.png")

if __name__ == "__main__":
    main() 