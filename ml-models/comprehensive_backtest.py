#!/usr/bin/env python3
"""
Comprehensive backtesting framework that tests the strategy under:
1. Different market conditions (bull/bear/sideways)
2. Different pricing scenarios (normal/volatile/trending)
3. Markowitz portfolio optimization
4. Risk-adjusted performance metrics
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

def classify_market_conditions(df, window=50):
    """Classify market conditions based on price action."""
    print("Classifying market conditions...")
    
    conditions = pd.DataFrame(index=df.index)
    conditions['datetime'] = df['datetime']
    conditions['pair'] = df['pair']
    
    for pair in df['pair'].unique():
        pair_mask = df['pair'] == pair
        pair_data = df[pair_mask].copy()
        
        # Calculate rolling metrics
        pair_data['rolling_return'] = pair_data['close'].pct_change(window)
        pair_data['rolling_volatility'] = pair_data['close'].pct_change().rolling(window).std()
        pair_data['rolling_trend'] = pair_data['close'].rolling(window).mean() - pair_data['close'].rolling(window*2).mean()
        
        # Classify market conditions
        # Bull market: positive trend, low volatility
        bull_condition = (pair_data['rolling_trend'] > 0) & (pair_data['rolling_volatility'] < pair_data['rolling_volatility'].quantile(0.7))
        
        # Bear market: negative trend, low volatility
        bear_condition = (pair_data['rolling_trend'] < 0) & (pair_data['rolling_volatility'] < pair_data['rolling_volatility'].quantile(0.7))
        
        # Volatile market: high volatility
        volatile_condition = pair_data['rolling_volatility'] > pair_data['rolling_volatility'].quantile(0.8)
        
        # Sideways market: low trend, low volatility
        sideways_condition = (np.abs(pair_data['rolling_trend']) < pair_data['rolling_trend'].std()) & (pair_data['rolling_volatility'] < pair_data['rolling_volatility'].quantile(0.6))
        
        # Assign conditions
        conditions.loc[pair_mask, 'market_condition'] = 'normal'
        conditions.loc[pair_mask & bull_condition, 'market_condition'] = 'bull'
        conditions.loc[pair_mask & bear_condition, 'market_condition'] = 'bear'
        conditions.loc[pair_mask & volatile_condition, 'market_condition'] = 'volatile'
        conditions.loc[pair_mask & sideways_condition, 'market_condition'] = 'sideways'
        
        # Add volatility and trend metrics
        conditions.loc[pair_mask, 'volatility'] = pair_data['rolling_volatility']
        conditions.loc[pair_mask, 'trend'] = pair_data['rolling_trend']
    
    return conditions

def calculate_returns_with_conditions(signals, market_conditions, slippage=0.002, max_position_size=0.1, confidence_threshold=0.002):
    """Calculate strategy returns with market condition analysis."""
    print("Calculating returns with market condition analysis...")
    
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
        
        # Calculate Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Calculate Sortino ratio
        negative_returns = strategy_return[strategy_return < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = np.mean(strategy_return) / downside_deviation if downside_deviation > 0 else 0
        
        # Store results
        results[strategy_name] = {
            'signal': signal,
            'strategy_return': strategy_return,
            'cumulative_return': cumulative_return,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'max_drawdown': max_drawdown,
            'position_size': position_size
        }
    
    return results

def analyze_by_market_condition(signals, results, market_conditions):
    """Analyze results by market condition."""
    print("Analyzing results by market condition...")
    
    condition_results = {}
    
    for condition in market_conditions['market_condition'].unique():
        condition_mask = market_conditions['market_condition'] == condition
        condition_signals = signals[condition_mask]
        
        condition_results[condition] = {}
        
        for strategy_name, strategy_data in results.items():
            condition_strategy_return = strategy_data['strategy_return'][condition_mask]
            condition_signal = strategy_data['signal'][condition_mask]
            
            # Calculate condition-specific metrics
            total_return = np.sum(condition_strategy_return)
            sharpe_ratio = np.mean(condition_strategy_return) / np.std(condition_strategy_return) if np.std(condition_strategy_return) > 0 else 0
            win_rate = np.mean(condition_strategy_return > 0) if np.sum(condition_signal != 0) > 0 else 0
            num_trades = np.sum(condition_signal != 0)
            
            # Calculate max drawdown for this condition
            cumulative_return = np.cumprod(1 + condition_strategy_return) - 1
            peak = np.maximum.accumulate(1 + cumulative_return)
            drawdown = (1 + cumulative_return) / peak - 1
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            
            condition_results[condition][strategy_name] = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'num_trades': num_trades,
                'max_drawdown': max_drawdown
            }
    
    return condition_results

def markowitz_optimization(returns_data, target_return=None, risk_free_rate=0.02):
    """Perform Markowitz portfolio optimization."""
    print("Performing Markowitz portfolio optimization...")
    
    # Calculate expected returns and covariance matrix
    expected_returns = np.mean(returns_data, axis=0)
    cov_matrix = np.cov(returns_data.T)
    
    n_assets = len(expected_returns)
    
    # Define objective function (minimize portfolio variance)
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return portfolio_variance
    
    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
    ]
    
    if target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return})
    
    # Bounds: weights between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weights
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    if result.success:
        optimal_weights = result.x
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_volatility = np.sqrt(result.fun)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
    else:
        print(f"Optimization failed: {result.message}")
        return None

def create_portfolio_returns(signals, results, market_conditions, rebalance_freq=20):
    """Create portfolio returns using Markowitz optimization."""
    print("Creating portfolio returns with Markowitz optimization...")
    
    # Get strategy names
    strategy_names = list(results.keys())
    
    # Prepare returns data for optimization
    returns_data = []
    for strategy_name in strategy_names:
        returns_data.append(results[strategy_name]['strategy_return'])
    
    returns_data = np.array(returns_data).T  # Shape: (time_periods, strategies)
    
    # Initialize portfolio
    portfolio_returns = []
    portfolio_weights = []
    
    # Rebalance portfolio every rebalance_freq periods
    for i in range(0, len(returns_data), rebalance_freq):
        end_idx = min(i + rebalance_freq, len(returns_data))
        
        # Use historical data for optimization
        if i > 0:
            historical_returns = returns_data[:i]
            
            # Perform optimization
            optimization_result = markowitz_optimization(historical_returns)
            
            if optimization_result is not None:
                weights = optimization_result['weights']
            else:
                # Use equal weights if optimization fails
                weights = np.array([1/len(strategy_names)] * len(strategy_names))
        else:
            # Equal weights for first period
            weights = np.array([1/len(strategy_names)] * len(strategy_names))
        
        # Calculate portfolio returns for this period
        period_returns = returns_data[i:end_idx]
        portfolio_period_returns = np.dot(period_returns, weights)
        
        portfolio_returns.extend(portfolio_period_returns)
        portfolio_weights.extend([weights] * len(portfolio_period_returns))
    
    # Calculate portfolio metrics
    portfolio_returns = np.array(portfolio_returns)
    cumulative_portfolio_return = np.cumprod(1 + portfolio_returns) - 1
    
    # Calculate metrics
    total_return = cumulative_portfolio_return[-1] if len(cumulative_portfolio_return) > 0 else 0
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
    max_drawdown = np.min((np.cumprod(1 + portfolio_returns) / np.maximum.accumulate(np.cumprod(1 + portfolio_returns))) - 1)
    
    return {
        'portfolio_returns': portfolio_returns,
        'cumulative_return': cumulative_portfolio_return,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'weights': portfolio_weights
    }

def plot_comprehensive_results(signals, results, market_conditions, portfolio_results):
    """Plot comprehensive backtesting results."""
    print("Creating comprehensive plots...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Subplot 1: Cumulative returns comparison
    ax1 = axes[0, 0]
    for strategy_name, strategy_data in results.items():
        ax1.plot(signals['datetime'], strategy_data['cumulative_return'], 
                label=f'{strategy_name} ({strategy_data["total_return"]:.3f})')
    
    if portfolio_results:
        ax1.plot(signals['datetime'][:len(portfolio_results['cumulative_return'])], 
                portfolio_results['cumulative_return'], 
                label=f'Markowitz Portfolio ({portfolio_results["total_return"]:.3f})', 
                linewidth=2, color='black')
    
    ax1.set_title('Cumulative Strategy Returns')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True)
    
    # Subplot 2: Market conditions distribution
    ax2 = axes[0, 1]
    condition_counts = market_conditions['market_condition'].value_counts()
    ax2.pie(condition_counts.values, labels=condition_counts.index, autopct='%1.1f%%')
    ax2.set_title('Market Conditions Distribution')
    
    # Subplot 3: Performance metrics comparison
    ax3 = axes[0, 2]
    metrics = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
    strategy_names = list(results.keys())
    
    x = np.arange(len(strategy_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in strategy_names]
        ax3.bar(x + i*width, values, width, label=metric)
    
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Value')
    ax3.set_title('Performance Metrics Comparison')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(strategy_names, rotation=45)
    ax3.legend()
    ax3.grid(True)
    
    # Subplot 4: Returns by market condition
    ax4 = axes[1, 0]
    condition_results = analyze_by_market_condition(signals, results, market_conditions)
    
    conditions = list(condition_results.keys())
    strategy_names = list(results.keys())
    
    x = np.arange(len(conditions))
    width = 0.2
    
    for i, strategy_name in enumerate(strategy_names):
        values = [condition_results[condition][strategy_name]['total_return'] 
                 for condition in conditions]
        ax4.bar(x + i*width, values, width, label=strategy_name)
    
    ax4.set_xlabel('Market Condition')
    ax4.set_ylabel('Total Return')
    ax4.set_title('Returns by Market Condition')
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(conditions, rotation=45)
    ax4.legend()
    ax4.grid(True)
    
    # Subplot 5: Sharpe ratios by market condition
    ax5 = axes[1, 1]
    for i, strategy_name in enumerate(strategy_names):
        values = [condition_results[condition][strategy_name]['sharpe_ratio'] 
                 for condition in conditions]
        ax5.bar(x + i*width, values, width, label=strategy_name)
    
    ax5.set_xlabel('Market Condition')
    ax5.set_ylabel('Sharpe Ratio')
    ax5.set_title('Sharpe Ratios by Market Condition')
    ax5.set_xticks(x + width * 1.5)
    ax5.set_xticklabels(conditions, rotation=45)
    ax5.legend()
    ax5.grid(True)
    
    # Subplot 6: Drawdown comparison
    ax6 = axes[1, 2]
    for strategy_name, strategy_data in results.items():
        cumulative_return = strategy_data['cumulative_return']
        peak = np.maximum.accumulate(1 + cumulative_return)
        drawdown = (1 + cumulative_return) / peak - 1
        ax6.plot(signals['datetime'], drawdown, label=f'{strategy_name}')
    
    if portfolio_results:
        portfolio_cumulative = portfolio_results['cumulative_return']
        portfolio_peak = np.maximum.accumulate(1 + portfolio_cumulative)
        portfolio_drawdown = (1 + portfolio_cumulative) / portfolio_peak - 1
        ax6.plot(signals['datetime'][:len(portfolio_drawdown)], portfolio_drawdown, 
                label='Markowitz Portfolio', linewidth=2, color='black')
    
    ax6.set_title('Strategy Drawdowns')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Drawdown')
    ax6.legend()
    ax6.grid(True)
    
    # Subplot 7: Volatility vs Return scatter
    ax7 = axes[2, 0]
    for strategy_name, strategy_data in results.items():
        volatility = np.std(strategy_data['strategy_return'])
        total_return = strategy_data['total_return']
        ax7.scatter(volatility, total_return, label=strategy_name, s=100)
    
    if portfolio_results:
        portfolio_volatility = np.std(portfolio_results['portfolio_returns'])
        portfolio_return = portfolio_results['total_return']
        ax7.scatter(portfolio_volatility, portfolio_return, 
                   label='Markowitz Portfolio', s=100, color='black', marker='*')
    
    ax7.set_xlabel('Volatility')
    ax7.set_ylabel('Total Return')
    ax7.set_title('Risk-Return Profile')
    ax7.legend()
    ax7.grid(True)
    
    # Subplot 8: Win rate by market condition
    ax8 = axes[2, 1]
    for i, strategy_name in enumerate(strategy_names):
        values = [condition_results[condition][strategy_name]['win_rate'] 
                 for condition in conditions]
        ax8.bar(x + i*width, values, width, label=strategy_name)
    
    ax8.set_xlabel('Market Condition')
    ax8.set_ylabel('Win Rate')
    ax8.set_title('Win Rate by Market Condition')
    ax8.set_xticks(x + width * 1.5)
    ax8.set_xticklabels(conditions, rotation=45)
    ax8.legend()
    ax8.grid(True)
    
    # Subplot 9: Portfolio weights evolution (if available)
    ax9 = axes[2, 2]
    if portfolio_results and len(portfolio_results['weights']) > 0:
        weights_data = np.array(portfolio_results['weights'])
        for i, strategy_name in enumerate(strategy_names):
            ax9.plot(weights_data[:, i], label=strategy_name)
        
        ax9.set_xlabel('Rebalance Period')
        ax9.set_ylabel('Weight')
        ax9.set_title('Portfolio Weights Evolution')
        ax9.legend()
        ax9.grid(True)
    else:
        ax9.text(0.5, 0.5, 'Portfolio weights\nnot available', 
                ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Portfolio Weights Evolution')
    
    plt.tight_layout()
    plt.savefig('comprehensive_backtest_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main comprehensive backtesting function."""
    print("🚀 Starting comprehensive backtest...")
    
    # Load models and data
    models, scaler, df, feature_cols = load_models_and_data()
    
    if not models:
        print("❌ No models loaded. Exiting.")
        return
    
    # Generate signals
    signals = generate_signals(df, models, scaler, feature_cols, threshold=0.001)
    
    # Classify market conditions
    market_conditions = classify_market_conditions(df)
    
    # Calculate returns with market condition analysis
    results = calculate_returns_with_conditions(signals, market_conditions)
    
    # Create portfolio returns using Markowitz optimization
    portfolio_results = create_portfolio_returns(signals, results, market_conditions)
    
    # Analyze by market condition
    condition_results = analyze_by_market_condition(signals, results, market_conditions)
    
    # Print comprehensive results
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE BACKTEST RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n📈 Overall Strategy Results:")
    for strategy_name, strategy_data in results.items():
        print(f"  {strategy_name}:")
        print(f"    Total Return: {strategy_data['total_return']:.3f}")
        print(f"    Sharpe Ratio: {strategy_data['sharpe_ratio']:.3f}")
        print(f"    Sortino Ratio: {strategy_data['sortino_ratio']:.3f}")
        print(f"    Calmar Ratio: {strategy_data['calmar_ratio']:.3f}")
        print(f"    Win Rate: {strategy_data['win_rate']:.3f}")
        print(f"    Number of Trades: {strategy_data['num_trades']}")
        print(f"    Max Drawdown: {strategy_data['max_drawdown']:.3f}")
    
    print(f"\n🏦 Markowitz Portfolio Results:")
    print(f"    Total Return: {portfolio_results['total_return']:.3f}")
    print(f"    Sharpe Ratio: {portfolio_results['sharpe_ratio']:.3f}")
    print(f"    Max Drawdown: {portfolio_results['max_drawdown']:.3f}")
    
    print(f"\n🌍 Market Conditions Analysis:")
    for condition, condition_data in condition_results.items():
        print(f"\n  {condition.upper()} Market:")
        for strategy_name, metrics in condition_data.items():
            print(f"    {strategy_name}: Return={metrics['total_return']:.3f}, "
                  f"Sharpe={metrics['sharpe_ratio']:.3f}, "
                  f"Win Rate={metrics['win_rate']:.3f}, "
                  f"Trades={metrics['num_trades']}")
    
    # Plot comprehensive results
    plot_comprehensive_results(signals, results, market_conditions, portfolio_results)
    
    # Save comprehensive results
    comprehensive_results = {
        'individual_strategies': results,
        'portfolio_results': portfolio_results,
        'market_conditions': condition_results
    }
    
    # Save to CSV
    results_df = pd.DataFrame({
        strategy_name: {
            'total_return': data['total_return'],
            'sharpe_ratio': data['sharpe_ratio'],
            'sortino_ratio': data['sortino_ratio'],
            'calmar_ratio': data['calmar_ratio'],
            'win_rate': data['win_rate'],
            'num_trades': data['num_trades'],
            'max_drawdown': data['max_drawdown']
        } for strategy_name, data in results.items()
    }).T
    
    results_df.to_csv('comprehensive_backtest_results.csv')
    print(f"\n✅ Results saved to comprehensive_backtest_results.csv")
    print(f"✅ Plots saved to comprehensive_backtest_results.png")

if __name__ == "__main__":
    main() 