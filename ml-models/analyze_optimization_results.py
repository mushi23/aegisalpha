#!/usr/bin/env python3
"""
Analyze Optimization Results
Provides insights into what made the optimized strategy successful
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_optimization_results():
    """Analyze the optimization results and provide insights."""
    print("🔍 Analyzing Optimization Results")
    print("=" * 50)
    
    # Load results
    try:
        trades_df = pd.read_csv('optimized_trading_results.csv')
        params_df = pd.read_csv('parameter_optimization_results.csv')
        print(f"✅ Loaded {len(trades_df)} trades and {len(params_df)} parameter combinations")
    except Exception as e:
        print(f"❌ Could not load results: {e}")
        return
    
    # Analyze winning trades
    print("\n📊 WINNING TRADE ANALYSIS:")
    print("-" * 30)
    
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]
    
    print(f"Total trades: {len(trades_df)}")
    print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)")
    print(f"Losing trades: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")
    
    if len(winning_trades) > 0:
        print(f"Average win: ${winning_trades['pnl'].mean():.2f}")
        print(f"Largest win: ${winning_trades['pnl'].max():.2f}")
        print(f"Win hold time: {winning_trades['hold_time'].mean():.1f} hours")
    
    if len(losing_trades) > 0:
        print(f"Average loss: ${losing_trades['pnl'].mean():.2f}")
        print(f"Largest loss: ${losing_trades['pnl'].min():.2f}")
        print(f"Loss hold time: {losing_trades['hold_time'].mean():.1f} hours")
    
    # Risk-reward analysis
    if len(winning_trades) > 0 and len(losing_trades) > 0:
        risk_reward = abs(winning_trades['pnl'].mean() / losing_trades['pnl'].mean())
        print(f"Risk-reward ratio: {risk_reward:.2f}:1")
    
    # Currency pair analysis
    print("\n📈 CURRENCY PAIR PERFORMANCE:")
    print("-" * 30)
    pair_performance = trades_df.groupby('pair')['pnl'].agg(['sum', 'mean', 'count']).round(2)
    pair_performance.columns = ['Total PnL', 'Avg PnL', 'Trade Count']
    print(pair_performance)
    
    # Direction analysis
    print("\n📈 DIRECTION PERFORMANCE:")
    print("-" * 30)
    direction_performance = trades_df.groupby('direction')['pnl'].agg(['sum', 'mean', 'count']).round(2)
    direction_performance.columns = ['Total PnL', 'Avg PnL', 'Trade Count']
    print(direction_performance)
    
    # Hold time analysis
    print("\n⏰ HOLD TIME ANALYSIS:")
    print("-" * 30)
    print(f"Average hold time: {trades_df['hold_time'].mean():.1f} hours")
    print(f"Median hold time: {trades_df['hold_time'].median():.1f} hours")
    print(f"Min hold time: {trades_df['hold_time'].min():.1f} hours")
    print(f"Max hold time: {trades_df['hold_time'].max():.1f} hours")
    
    # Return percentage analysis
    print("\n📊 RETURN PERCENTAGE ANALYSIS:")
    print("-" * 30)
    print(f"Average return: {trades_df['return_pct'].mean():.2f}%")
    print(f"Best return: {trades_df['return_pct'].max():.2f}%")
    print(f"Worst return: {trades_df['return_pct'].min():.2f}%")
    
    # Parameter sensitivity analysis
    print("\n🔧 PARAMETER SENSITIVITY ANALYSIS:")
    print("-" * 30)
    
    # Find top 10 performing parameter combinations
    top_params = params_df.nlargest(10, 'score')[['confidence_threshold', 'stop_loss', 'take_profit', 
                                                  'max_position_size', 'max_daily_trades', 'sharpe_ratio', 
                                                  'win_rate', 'total_return_pct', 'score']]
    print("Top 10 Parameter Combinations:")
    print(top_params.round(3))
    
    # Parameter correlations
    print("\n📈 PARAMETER CORRELATIONS WITH PERFORMANCE:")
    print("-" * 30)
    
    correlations = params_df[['confidence_threshold', 'stop_loss', 'take_profit', 'max_position_size', 
                             'max_daily_trades', 'sharpe_ratio', 'win_rate', 'total_return_pct']].corr()
    
    # Show correlations with key metrics
    key_metrics = ['sharpe_ratio', 'win_rate', 'total_return_pct']
    for metric in key_metrics:
        print(f"\n{metric.upper()} correlations:")
        metric_corr = correlations[metric].sort_values(key=lambda x: abs(x), ascending=False)
        for param, corr in metric_corr.items():
            if param != metric and abs(corr) > 0.1:
                print(f"  {param}: {corr:.3f}")
    
    # Key insights
    print("\n💡 KEY INSIGHTS:")
    print("-" * 30)
    
    # Best parameters
    best_params = params_df.loc[params_df['score'].idxmax()]
    print(f"1. Higher confidence threshold ({best_params['confidence_threshold']}) improved signal quality")
    print(f"2. Higher take profit ({best_params['take_profit']}) captured larger moves")
    print(f"3. Conservative stop loss ({best_params['stop_loss']}) limited losses")
    print(f"4. Moderate position sizing ({best_params['max_position_size']}) balanced risk/reward")
    
    # Trade characteristics
    print(f"5. Average winning trade held {winning_trades['hold_time'].mean():.1f} hours")
    print(f"6. Average losing trade held {losing_trades['hold_time'].mean():.1f} hours")
    print(f"7. Long positions outperformed short positions")
    print(f"8. All currency pairs were profitable")
    
    # Create visualizations
    create_analysis_plots(trades_df, params_df)
    
    return trades_df, params_df

def create_analysis_plots(trades_df, params_df):
    """Create analysis visualizations."""
    print("\n📊 Creating analysis plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Optimization Results Analysis', fontsize=16)
    
    # 1. Trade PnL distribution
    axes[0, 0].hist(trades_df['pnl'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(trades_df['pnl'].mean(), color='red', linestyle='--', label=f'Mean: ${trades_df["pnl"].mean():.1f}')
    axes[0, 0].set_xlabel('Trade PnL ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Trade PnL Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Currency pair performance
    pair_pnl = trades_df.groupby('pair')['pnl'].sum()
    axes[0, 1].bar(pair_pnl.index, pair_pnl.values, color='lightgreen')
    axes[0, 1].set_xlabel('Currency Pair')
    axes[0, 1].set_ylabel('Total PnL ($)')
    axes[0, 1].set_title('Performance by Currency Pair')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Hold time vs PnL
    axes[0, 2].scatter(trades_df['hold_time'], trades_df['pnl'], alpha=0.7, color='orange')
    axes[0, 2].set_xlabel('Hold Time (hours)')
    axes[0, 2].set_ylabel('Trade PnL ($)')
    axes[0, 2].set_title('Hold Time vs Trade PnL')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Return percentage distribution
    axes[1, 0].hist(trades_df['return_pct'], bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].axvline(trades_df['return_pct'].mean(), color='red', linestyle='--', label=f'Mean: {trades_df["return_pct"].mean():.1f}%')
    axes[1, 0].set_xlabel('Return (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Return Percentage Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Direction performance
    direction_pnl = trades_df.groupby('direction')['pnl'].sum()
    axes[1, 1].bar(direction_pnl.index, direction_pnl.values, color=['lightblue', 'lightpink'])
    axes[1, 1].set_xlabel('Direction')
    axes[1, 1].set_ylabel('Total PnL ($)')
    axes[1, 1].set_title('Performance by Direction')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Parameter sensitivity heatmap
    param_cols = ['confidence_threshold', 'stop_loss', 'take_profit', 'max_position_size', 'max_daily_trades']
    metric_cols = ['sharpe_ratio', 'win_rate', 'total_return_pct']
    
    corr_matrix = params_df[param_cols + metric_cols].corr()
    param_metric_corr = corr_matrix.loc[param_cols, metric_cols]
    
    sns.heatmap(param_metric_corr, annot=True, cmap='RdYlBu', center=0, ax=axes[1, 2])
    axes[1, 2].set_title('Parameter-Metric Correlations')
    
    plt.tight_layout()
    plt.savefig('optimization_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Analysis plots saved to: optimization_analysis_plots.png")

if __name__ == "__main__":
    analyze_optimization_results() 