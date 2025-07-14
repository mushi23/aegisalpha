#!/usr/bin/env python3
"""
Compare Original vs Enhanced Backtest Results
Analyzes the differences and improvements between the two approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_compare_results():
    """Load and compare original vs enhanced backtest results."""
    print("🔄 Loading and comparing backtest results...")
    
    # Load original results
    try:
        original_results = pd.read_csv('realistic_backtest_results.csv', index_col=0)
        print("✅ Loaded original results")
    except FileNotFoundError:
        print("❌ Original results file not found")
        return None, None
    
    # Load enhanced results
    try:
        enhanced_results = pd.read_csv('enhanced_realistic_backtest_results.csv', index_col=0)
        print("✅ Loaded enhanced results")
    except FileNotFoundError:
        print("❌ Enhanced results file not found")
        return original_results, None
    
    return original_results, enhanced_results

def analyze_improvements(original_results, enhanced_results):
    """Analyze improvements between original and enhanced results."""
    print("\n📊 Analyzing improvements...")
    
    if enhanced_results is None:
        print("⚠️ No enhanced results to compare")
        return
    
    # Calculate improvements
    improvements = {}
    
    for strategy in original_results.index:
        if strategy in enhanced_results.index:
            orig = original_results.loc[strategy]
            enh = enhanced_results.loc[strategy]
            
            improvements[strategy] = {
                'total_return_improvement': (enh['total_return'] - orig['total_return']) / abs(orig['total_return']) * 100 if orig['total_return'] != 0 else 0,
                'sharpe_improvement': (enh['sharpe_ratio'] - orig['sharpe_ratio']) / abs(orig['sharpe_ratio']) * 100 if orig['sharpe_ratio'] != 0 else 0,
                'win_rate_improvement': (enh['win_rate'] - orig['win_rate']) / orig['win_rate'] * 100 if orig['win_rate'] != 0 else 0,
                'max_drawdown_improvement': (orig['max_drawdown'] - enh['max_drawdown']) / abs(orig['max_drawdown']) * 100 if orig['max_drawdown'] != 0 else 0,
                'trades_change': enh['num_trades'] - orig['num_trades']
            }
    
    return improvements

def plot_comparison(original_results, enhanced_results, improvements):
    """Plot comparison between original and enhanced results."""
    print("🔄 Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Subplot 1: Total Return Comparison
    ax1 = axes[0, 0]
    strategies = list(original_results.index)
    x = np.arange(len(strategies))
    width = 0.35
    
    orig_returns = [original_results.loc[s, 'total_return'] for s in strategies]
    enh_returns = [enhanced_results.loc[s, 'total_return'] for s in strategies] if enhanced_results is not None else [0] * len(strategies)
    
    ax1.bar(x - width/2, orig_returns, width, label='Original', alpha=0.8)
    if enhanced_results is not None:
        ax1.bar(x + width/2, enh_returns, width, label='Enhanced', alpha=0.8)
    
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Total Return')
    ax1.set_title('Total Return Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Sharpe Ratio Comparison
    ax2 = axes[0, 1]
    orig_sharpe = [original_results.loc[s, 'sharpe_ratio'] for s in strategies]
    enh_sharpe = [enhanced_results.loc[s, 'sharpe_ratio'] for s in strategies] if enhanced_results is not None else [0] * len(strategies)
    
    ax2.bar(x - width/2, orig_sharpe, width, label='Original', alpha=0.8)
    if enhanced_results is not None:
        ax2.bar(x + width/2, enh_sharpe, width, label='Enhanced', alpha=0.8)
    
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Win Rate Comparison
    ax3 = axes[0, 2]
    orig_winrate = [original_results.loc[s, 'win_rate'] for s in strategies]
    enh_winrate = [enhanced_results.loc[s, 'win_rate'] for s in strategies] if enhanced_results is not None else [0] * len(strategies)
    
    ax3.bar(x - width/2, orig_winrate, width, label='Original', alpha=0.8)
    if enhanced_results is not None:
        ax3.bar(x + width/2, enh_winrate, width, label='Enhanced', alpha=0.8)
    
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Win Rate')
    ax3.set_title('Win Rate Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Max Drawdown Comparison
    ax4 = axes[1, 0]
    orig_dd = [original_results.loc[s, 'max_drawdown'] for s in strategies]
    enh_dd = [enhanced_results.loc[s, 'max_drawdown'] for s in strategies] if enhanced_results is not None else [0] * len(strategies)
    
    ax4.bar(x - width/2, orig_dd, width, label='Original', alpha=0.8, color='red')
    if enhanced_results is not None:
        ax4.bar(x + width/2, enh_dd, width, label='Enhanced', alpha=0.8, color='orange')
    
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Max Drawdown')
    ax4.set_title('Max Drawdown Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Subplot 5: Number of Trades Comparison
    ax5 = axes[1, 1]
    orig_trades = [original_results.loc[s, 'num_trades'] for s in strategies]
    enh_trades = [enhanced_results.loc[s, 'num_trades'] for s in strategies] if enhanced_results is not None else [0] * len(strategies)
    
    ax5.bar(x - width/2, orig_trades, width, label='Original', alpha=0.8)
    if enhanced_results is not None:
        ax5.bar(x + width/2, enh_trades, width, label='Enhanced', alpha=0.8)
    
    ax5.set_xlabel('Strategy')
    ax5.set_ylabel('Number of Trades')
    ax5.set_title('Number of Trades Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(strategies, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Improvement Summary
    ax6 = axes[1, 2]
    if improvements:
        improvement_metrics = ['total_return_improvement', 'sharpe_improvement', 'win_rate_improvement']
        improvement_data = []
        strategy_names = []
        
        for strategy, imp in improvements.items():
            strategy_names.append(strategy)
            improvement_data.append([imp[metric] for metric in improvement_metrics])
        
        improvement_data = np.array(improvement_data)
        
        x = np.arange(len(strategy_names))
        width = 0.25
        
        for i, metric in enumerate(improvement_metrics):
            ax6.bar(x + i*width, improvement_data[:, i], width, label=metric.replace('_', ' ').title())
        
        ax6.set_xlabel('Strategy')
        ax6.set_ylabel('Improvement (%)')
        ax6.set_title('Improvement Summary')
        ax6.set_xticks(x + width)
        ax6.set_xticklabels(strategy_names, rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_comparison(original_results, enhanced_results, improvements):
    """Print detailed comparison analysis."""
    print("\n" + "="*80)
    print("📊 DETAILED BACKTEST COMPARISON ANALYSIS")
    print("="*80)
    
    print(f"\n📈 Original Results Summary:")
    print(original_results.round(4))
    
    if enhanced_results is not None:
        print(f"\n🚀 Enhanced Results Summary:")
        print(enhanced_results.round(4))
        
        print(f"\n📊 Key Improvements:")
        for strategy, imp in improvements.items():
            print(f"\n  {strategy}:")
            print(f"    Total Return: {imp['total_return_improvement']:+.2f}%")
            print(f"    Sharpe Ratio: {imp['sharpe_improvement']:+.2f}%")
            print(f"    Win Rate: {imp['win_rate_improvement']:+.2f}%")
            print(f"    Max Drawdown: {imp['max_drawdown_improvement']:+.2f}%")
            print(f"    Trades Change: {imp['trades_change']:+.0f}")
        
        # Find best performing strategy
        best_original = original_results['sharpe_ratio'].idxmax()
        best_enhanced = enhanced_results['sharpe_ratio'].idxmax()
        
        print(f"\n🏆 Best Strategies:")
        print(f"  Original: {best_original} (Sharpe: {original_results.loc[best_original, 'sharpe_ratio']:.4f})")
        print(f"  Enhanced: {best_enhanced} (Sharpe: {enhanced_results.loc[best_enhanced, 'sharpe_ratio']:.4f})")
        
        # Overall improvement
        avg_sharpe_improvement = np.mean([imp['sharpe_improvement'] for imp in improvements.values()])
        avg_return_improvement = np.mean([imp['total_return_improvement'] for imp in improvements.values()])
        
        print(f"\n📈 Overall Improvements:")
        print(f"  Average Sharpe Ratio Improvement: {avg_sharpe_improvement:+.2f}%")
        print(f"  Average Return Improvement: {avg_return_improvement:+.2f}%")

def main():
    """Main function to run comparison analysis."""
    print("🚀 Starting backtest results comparison...")
    
    # Load results
    original_results, enhanced_results = load_and_compare_results()
    
    if original_results is None:
        print("❌ No results to compare. Exiting.")
        return
    
    # Analyze improvements
    improvements = analyze_improvements(original_results, enhanced_results)
    
    # Print detailed comparison
    print_detailed_comparison(original_results, enhanced_results, improvements)
    
    # Create comparison plots
    plot_comparison(original_results, enhanced_results, improvements)
    
    print(f"\n✅ Comparison analysis completed!")
    print(f"📁 Output file: backtest_comparison.png")

if __name__ == "__main__":
    main() 