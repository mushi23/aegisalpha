#!/usr/bin/env python3
"""
Profit Analysis
Comprehensive analysis of trading profits and performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_profits():
    """Analyze profits from both optimization and validation tests."""
    print("💰 PROFIT ANALYSIS")
    print("=" * 60)
    
    # Load both datasets
    try:
        opt_trades = pd.read_csv('optimized_trading_results.csv')
        val_trades = pd.read_csv('validation_test_results.csv')
        print(f"✅ Loaded {len(opt_trades)} optimization trades and {len(val_trades)} validation trades")
    except Exception as e:
        print(f"❌ Could not load results: {e}")
        return
    
    # Optimization Results Analysis
    print("\n📊 OPTIMIZATION PERIOD PROFITS (3 Months):")
    print("-" * 50)
    analyze_period_profits(opt_trades, "Optimization")
    
    # Validation Results Analysis
    print("\n📊 VALIDATION PERIOD PROFITS (12 Months):")
    print("-" * 50)
    analyze_period_profits(val_trades, "Validation")
    
    # Combined Analysis
    print("\n📊 COMBINED PROFIT ANALYSIS:")
    print("-" * 50)
    combined_trades = pd.concat([opt_trades, val_trades], ignore_index=True)
    analyze_period_profits(combined_trades, "Combined")
    
    # Create profit visualizations
    create_profit_visualizations(opt_trades, val_trades, combined_trades)
    
    return opt_trades, val_trades, combined_trades

def analyze_period_profits(trades_df, period_name):
    """Analyze profits for a specific period."""
    if trades_df.empty:
        print(f"No trades found for {period_name}")
        return
    
    # Basic profit metrics
    total_pnl = trades_df['pnl'].sum()
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = winning_trades / total_trades * 100
    
    print(f"💰 Total P&L: ${total_pnl:.2f}")
    print(f"🔄 Total Trades: {total_trades}")
    print(f"✅ Winning Trades: {winning_trades} ({win_rate:.1f}%)")
    print(f"❌ Losing Trades: {losing_trades} ({100-win_rate:.1f}%)")
    
    # Profit breakdown
    if winning_trades > 0:
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
        max_win = trades_df[trades_df['pnl'] > 0]['pnl'].max()
        total_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        print(f"📈 Average Win: ${avg_win:.2f}")
        print(f"📈 Largest Win: ${max_win:.2f}")
        print(f"📈 Total Wins: ${total_wins:.2f}")
    
    if losing_trades > 0:
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
        max_loss = trades_df[trades_df['pnl'] < 0]['pnl'].min()
        total_losses = trades_df[trades_df['pnl'] < 0]['pnl'].sum()
        print(f"📉 Average Loss: ${avg_loss:.2f}")
        print(f"📉 Largest Loss: ${max_loss:.2f}")
        print(f"📉 Total Losses: ${total_losses:.2f}")
    
    # Risk metrics
    if winning_trades > 0 and losing_trades > 0:
        risk_reward = abs(avg_win / avg_loss)
        profit_factor = abs(total_wins / total_losses)
        print(f"⚖️  Risk-Reward Ratio: {risk_reward:.2f}:1")
        print(f"💹 Profit Factor: {profit_factor:.2f}")
    
    # Return analysis
    avg_return_pct = trades_df['return_pct'].mean()
    best_return = trades_df['return_pct'].max()
    worst_return = trades_df['return_pct'].min()
    print(f"📊 Average Return: {avg_return_pct:.2f}%")
    print(f"📊 Best Return: {best_return:.2f}%")
    print(f"📊 Worst Return: {worst_return:.2f}%")
    
    # Currency pair performance
    print(f"\n📈 PERFORMANCE BY CURRENCY PAIR:")
    pair_perf = trades_df.groupby('pair').agg({
        'pnl': ['sum', 'mean', 'count'],
        'return_pct': 'mean'
    }).round(2)
    pair_perf.columns = ['Total PnL', 'Avg PnL', 'Trade Count', 'Avg Return %']
    print(pair_perf)
    
    # Direction performance
    print(f"\n📈 PERFORMANCE BY DIRECTION:")
    dir_perf = trades_df.groupby('direction').agg({
        'pnl': ['sum', 'mean', 'count'],
        'return_pct': 'mean'
    }).round(2)
    dir_perf.columns = ['Total PnL', 'Avg PnL', 'Trade Count', 'Avg Return %']
    print(dir_perf)
    
    # Hold time analysis
    avg_hold_time = trades_df['hold_time'].mean()
    print(f"\n⏰ Average Hold Time: {avg_hold_time:.1f} hours ({avg_hold_time/24:.1f} days)")
    
    # Monthly performance (if we have date data)
    if 'entry_time' in trades_df.columns:
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
        monthly_perf = trades_df.groupby('month')['pnl'].sum().round(2)
        print(f"\n📅 MONTHLY PERFORMANCE:")
        for month, pnl in monthly_perf.items():
            print(f"  {month}: ${pnl:.2f}")

def create_profit_visualizations(opt_trades, val_trades, combined_trades):
    """Create comprehensive profit visualizations."""
    print("\n📊 Creating profit visualizations...")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Comprehensive Profit Analysis', fontsize=16)
    
    # 1. PnL Distribution - Optimization
    axes[0, 0].hist(opt_trades['pnl'], bins=10, alpha=0.7, color='green', edgecolor='black')
    axes[0, 0].axvline(opt_trades['pnl'].mean(), color='red', linestyle='--', 
                       label=f'Mean: ${opt_trades["pnl"].mean():.1f}')
    axes[0, 0].set_xlabel('Trade PnL ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Optimization Period - PnL Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. PnL Distribution - Validation
    axes[0, 1].hist(val_trades['pnl'], bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].axvline(val_trades['pnl'].mean(), color='red', linestyle='--', 
                       label=f'Mean: ${val_trades["pnl"].mean():.1f}')
    axes[0, 1].set_xlabel('Trade PnL ($)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Validation Period - PnL Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Cumulative PnL over time
    if 'entry_time' in combined_trades.columns:
        combined_trades['entry_time'] = pd.to_datetime(combined_trades['entry_time'])
        combined_trades = combined_trades.sort_values('entry_time')
        cumulative_pnl = combined_trades['pnl'].cumsum()
        
        axes[0, 2].plot(combined_trades['entry_time'], cumulative_pnl, linewidth=2, color='purple')
        axes[0, 2].set_xlabel('Date')
        axes[0, 2].set_ylabel('Cumulative PnL ($)')
        axes[0, 2].set_title('Cumulative PnL Over Time')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Currency pair performance comparison
    opt_pair_pnl = opt_trades.groupby('pair')['pnl'].sum()
    val_pair_pnl = val_trades.groupby('pair')['pnl'].sum()
    
    x = np.arange(len(opt_pair_pnl))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, opt_pair_pnl.values, width, label='Optimization', alpha=0.7, color='green')
    axes[1, 0].bar(x + width/2, val_pair_pnl.values, width, label='Validation', alpha=0.7, color='blue')
    axes[1, 0].set_xlabel('Currency Pair')
    axes[1, 0].set_ylabel('Total PnL ($)')
    axes[1, 0].set_title('Performance by Currency Pair')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(opt_pair_pnl.index, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Win/Loss ratio comparison
    opt_win_rate = len(opt_trades[opt_trades['pnl'] > 0]) / len(opt_trades) * 100
    val_win_rate = len(val_trades[val_trades['pnl'] > 0]) / len(val_trades) * 100
    
    periods = ['Optimization', 'Validation']
    win_rates = [opt_win_rate, val_win_rate]
    
    bars = axes[1, 1].bar(periods, win_rates, color=['green', 'blue'], alpha=0.7)
    axes[1, 1].set_ylabel('Win Rate (%)')
    axes[1, 1].set_title('Win Rate Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom')
    
    # 6. Return percentage distribution
    axes[1, 2].hist(combined_trades['return_pct'], bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 2].axvline(combined_trades['return_pct'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {combined_trades["return_pct"].mean():.1f}%')
    axes[1, 2].set_xlabel('Return (%)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Return Percentage Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Hold time vs PnL scatter
    axes[2, 0].scatter(combined_trades['hold_time'], combined_trades['pnl'], alpha=0.6, color='purple')
    axes[2, 0].set_xlabel('Hold Time (hours)')
    axes[2, 0].set_ylabel('Trade PnL ($)')
    axes[2, 0].set_title('Hold Time vs Trade PnL')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Direction performance
    dir_pnl = combined_trades.groupby('direction')['pnl'].sum()
    axes[2, 1].bar(dir_pnl.index, dir_pnl.values, color=['lightblue', 'lightpink'], alpha=0.7)
    axes[2, 1].set_xlabel('Direction')
    axes[2, 1].set_ylabel('Total PnL ($)')
    axes[2, 1].set_title('Performance by Direction')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Monthly performance heatmap
    if 'entry_time' in combined_trades.columns:
        combined_trades['month'] = combined_trades['entry_time'].dt.to_period('M')
        monthly_pair_pnl = combined_trades.pivot_table(
            index='month', columns='pair', values='pnl', aggfunc='sum'
        ).fillna(0)
        
        sns.heatmap(monthly_pair_pnl, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=axes[2, 2])
        axes[2, 2].set_title('Monthly Performance Heatmap')
        axes[2, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('comprehensive_profit_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Profit analysis charts saved to: comprehensive_profit_analysis.png")

def print_profit_summary():
    """Print a summary of key profit metrics."""
    print("\n" + "=" * 60)
    print("💰 PROFIT SUMMARY")
    print("=" * 60)
    
    try:
        opt_trades = pd.read_csv('optimized_trading_results.csv')
        val_trades = pd.read_csv('validation_test_results.csv')
        
        # Calculate key metrics
        opt_total_pnl = opt_trades['pnl'].sum()
        val_total_pnl = val_trades['pnl'].sum()
        combined_pnl = opt_total_pnl + val_total_pnl
        
        opt_win_rate = len(opt_trades[opt_trades['pnl'] > 0]) / len(opt_trades) * 100
        val_win_rate = len(val_trades[val_trades['pnl'] > 0]) / len(val_trades) * 100
        
        print(f"📊 OPTIMIZATION PERIOD (3 months):")
        print(f"   Total P&L: ${opt_total_pnl:.2f}")
        print(f"   Win Rate: {opt_win_rate:.1f}%")
        print(f"   Trades: {len(opt_trades)}")
        
        print(f"\n📊 VALIDATION PERIOD (12 months):")
        print(f"   Total P&L: ${val_total_pnl:.2f}")
        print(f"   Win Rate: {val_win_rate:.1f}%")
        print(f"   Trades: {len(val_trades)}")
        
        print(f"\n📊 COMBINED RESULTS:")
        print(f"   Total P&L: ${combined_pnl:.2f}")
        print(f"   Average P&L per trade: ${combined_pnl/(len(opt_trades)+len(val_trades)):.2f}")
        print(f"   Total Trades: {len(opt_trades) + len(val_trades)}")
        
        # Annualized return (assuming $10,000 initial capital)
        initial_capital = 10000
        total_return_pct = (combined_pnl / initial_capital) * 100
        print(f"   Total Return: {total_return_pct:.2f}%")
        
        # Best and worst trades
        all_trades = pd.concat([opt_trades, val_trades], ignore_index=True)
        best_trade = all_trades.loc[all_trades['pnl'].idxmax()]
        worst_trade = all_trades.loc[all_trades['pnl'].idxmin()]
        
        print(f"\n🏆 BEST TRADE:")
        print(f"   Pair: {best_trade['pair']}, Direction: {best_trade['direction']}")
        print(f"   P&L: ${best_trade['pnl']:.2f} ({best_trade['return_pct']:.2f}%)")
        print(f"   Hold Time: {best_trade['hold_time']:.1f} hours")
        
        print(f"\n💥 WORST TRADE:")
        print(f"   Pair: {worst_trade['pair']}, Direction: {worst_trade['direction']}")
        print(f"   P&L: ${worst_trade['pnl']:.2f} ({worst_trade['return_pct']:.2f}%)")
        print(f"   Hold Time: {worst_trade['hold_time']:.1f} hours")
        
    except Exception as e:
        print(f"❌ Error calculating summary: {e}")

if __name__ == "__main__":
    analyze_profits()
    print_profit_summary() 