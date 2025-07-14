#!/usr/bin/env python3
"""
Cost-Aware Signal Threshold Optimizer
Finds the optimal signal threshold that maximizes net returns after transaction costs and slippage.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CostAwareThresholdOptimizer:
    def __init__(self, transaction_cost=0.002, slippage=0.005):
        """
        Initialize cost-aware threshold optimizer
        
        Args:
            transaction_cost: Fixed transaction cost as decimal
            slippage: Slippage per trade as decimal
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.total_cost_per_trade = transaction_cost + slippage
        
    def load_data(self, data_path):
        """Load enhanced dataset with model predictions"""
        print("🔄 Loading data for threshold optimization...")
        
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # Create label if needed
        if 'label' not in data.columns and 'return' in data.columns:
            cost_per_trade = 0.002 + 0.005
            data['label'] = ((data['return'] - cost_per_trade) > 0).astype(int)
        
        print(f"✅ Loaded data: {data.shape}")
        print(f"   Date range: {data['datetime'].min()} to {data['datetime'].max()}")
        print(f"   Currency pairs: {data['pair'].unique()}")
        
        return data
    
    def generate_signals_for_threshold(self, data, threshold):
        """Generate signals for a given threshold using the tuned model"""
        print(f"  Testing threshold: {threshold:.3f}")
        
        # Load tuned model
        import joblib
        model = joblib.load('lgbm_best_model.pkl')
        
        # Load feature list
        with open('feature_list_available.txt', 'r') as f:
            features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        features = [f for f in features if f in data.columns]
        
        all_results = []
        
        for pair in data['pair'].unique():
            # Get data for this pair
            pair_data = data[data['pair'] == pair].copy()
            pair_data = pair_data.sort_values('datetime').reset_index(drop=True)
            
            # Prepare features and drop NaN
            X = pair_data[features].dropna()
            if len(X) == 0:
                continue
                
            # Get the indices of non-NaN rows
            valid_indices = X.index
            pair_data_clean = pair_data.loc[valid_indices].copy()
            
            # Train-test split (same as training)
            from sklearn.model_selection import train_test_split
            y = pair_data_clean['label']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False, stratify=None
            )
            
            # Generate signals for test set
            y_proba = model.predict_proba(X_test, predict_disable_shape_check=True)[:, 1]
            signals = (y_proba > threshold).astype(int)
            
            # Store results
            test_data = pair_data_clean.loc[X_test.index].copy()
            test_data['signal'] = signals
            test_data['signal_proba'] = y_proba
            
            all_results.append(test_data)
        
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            return combined_results
        else:
            return None
    
    def calculate_net_returns(self, data):
        """Calculate net returns after transaction costs and slippage"""
        # Ensure we have return column
        if 'return' not in data.columns and 'close' in data.columns:
            data['return'] = data['close'].pct_change()
        
        # Calculate gross strategy returns
        data['gross_return'] = data['signal'] * data['return']
        
        # Calculate transaction costs and slippage
        trade_change = data['signal'].diff().abs()  # 1 on entry/exit
        total_cost = trade_change * self.total_cost_per_trade
        data['net_return'] = data['gross_return'] - total_cost
        
        return data
    
    def calculate_performance_metrics(self, data):
        """Calculate comprehensive performance metrics"""
        # Annualize returns (assuming 4-hour data, 6 observations per day)
        annual_factor = 6 * 252
        
        # Net returns metrics
        net_returns = data['net_return']
        gross_returns = data['gross_return']
        
        # Basic metrics
        total_net_return = (1 + net_returns).prod() - 1
        annual_net_return = net_returns.mean() * annual_factor
        annual_volatility = net_returns.std() * np.sqrt(annual_factor)
        sharpe_ratio = annual_net_return / annual_volatility if annual_volatility > 0 else 0
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(net_returns)
        var_95 = np.percentile(net_returns, 5)  # 95% VaR
        
        # Trade metrics
        trade_frequency = data['signal'].mean()
        num_trades = (data['signal'].diff().abs() == 1).sum()
        
        # Cost metrics
        total_costs = (data['signal'].diff().abs() * self.total_cost_per_trade).sum()
        gross_return_sum = gross_returns.sum()
        net_return_sum = net_returns.sum()
        cost_drag = total_costs / gross_return_sum if gross_return_sum > 0 else 0
        
        # Win rate and profit factor
        win_rate = (net_returns > 0).mean()
        gross_profit = net_returns[net_returns > 0].sum()
        gross_loss = abs(net_returns[net_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_net_return': total_net_return,
            'annual_net_return': annual_net_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'trade_frequency': trade_frequency,
            'num_trades': num_trades,
            'total_costs': total_costs,
            'cost_drag': cost_drag,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def optimize_threshold(self, data_path, threshold_range=(0.5, 0.95), num_thresholds=20):
        """Find optimal threshold that maximizes net returns"""
        print("🚀 Starting cost-aware threshold optimization...")
        
        # Load data
        data = self.load_data(data_path)
        
        # Generate threshold range
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        
        results = []
        
        for threshold in thresholds:
            print(f"\n🔄 Testing threshold: {threshold:.3f}")
            
            # Generate signals for this threshold
            signals_data = self.generate_signals_for_threshold(data, threshold)
            
            if signals_data is None or len(signals_data) == 0:
                print(f"  ⚠️ No valid signals for threshold {threshold:.3f}")
                continue
            
            # Calculate net returns
            signals_data = self.calculate_net_returns(signals_data)
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(signals_data)
            metrics['threshold'] = threshold
            
            results.append(metrics)
            
            print(f"  ✅ Net Return: {metrics['annual_net_return']:.4f}, "
                  f"Sharpe: {metrics['sharpe_ratio']:.3f}, "
                  f"Trades: {metrics['num_trades']}, "
                  f"Cost Drag: {metrics['cost_drag']:.3f}")
        
        if not results:
            print("❌ No valid results generated!")
            return None
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold by different criteria
        optimal_by_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        optimal_by_return = results_df.loc[results_df['annual_net_return'].idxmax()]
        optimal_by_profit_factor = results_df.loc[results_df['profit_factor'].idxmax()]
        
        print(f"\n🎯 Optimal Thresholds:")
        print(f"   By Sharpe Ratio: {optimal_by_sharpe['threshold']:.3f} "
              f"(Sharpe: {optimal_by_sharpe['sharpe_ratio']:.3f})")
        print(f"   By Annual Return: {optimal_by_return['threshold']:.3f} "
              f"(Return: {optimal_by_return['annual_net_return']:.4f})")
        print(f"   By Profit Factor: {optimal_by_profit_factor['threshold']:.3f} "
              f"(PF: {optimal_by_profit_factor['profit_factor']:.3f})")
        
        return results_df, optimal_by_sharpe, optimal_by_return, optimal_by_profit_factor
    
    def plot_optimization_results(self, results_df, save_path="threshold_optimization_results.png"):
        """Plot threshold optimization results"""
        print("🔄 Creating optimization plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Sharpe Ratio vs Threshold
        ax1 = axes[0, 0]
        ax1.plot(results_df['threshold'], results_df['sharpe_ratio'], 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Signal Threshold')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('Sharpe Ratio vs Threshold')
        ax1.grid(True, alpha=0.3)
        
        # Mark optimal
        optimal_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        ax1.axvline(x=optimal_sharpe['threshold'], color='red', linestyle='--', alpha=0.7)
        ax1.text(optimal_sharpe['threshold'], optimal_sharpe['sharpe_ratio'], 
                f'Optimal: {optimal_sharpe["threshold"]:.3f}', 
                rotation=90, verticalalignment='bottom')
        
        # 2. Annual Return vs Threshold
        ax2 = axes[0, 1]
        ax2.plot(results_df['threshold'], results_df['annual_net_return'], 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Signal Threshold')
        ax2.set_ylabel('Annual Net Return')
        ax2.set_title('Annual Net Return vs Threshold')
        ax2.grid(True, alpha=0.3)
        
        # Mark optimal
        optimal_return = results_df.loc[results_df['annual_net_return'].idxmax()]
        ax2.axvline(x=optimal_return['threshold'], color='red', linestyle='--', alpha=0.7)
        ax2.text(optimal_return['threshold'], optimal_return['annual_net_return'], 
                f'Optimal: {optimal_return["threshold"]:.3f}', 
                rotation=90, verticalalignment='bottom')
        
        # 3. Profit Factor vs Threshold
        ax3 = axes[0, 2]
        ax3.plot(results_df['threshold'], results_df['profit_factor'], 'm-o', linewidth=2, markersize=6)
        ax3.set_xlabel('Signal Threshold')
        ax3.set_ylabel('Profit Factor')
        ax3.set_title('Profit Factor vs Threshold')
        ax3.grid(True, alpha=0.3)
        
        # Mark optimal
        optimal_pf = results_df.loc[results_df['profit_factor'].idxmax()]
        ax3.axvline(x=optimal_pf['threshold'], color='red', linestyle='--', alpha=0.7)
        ax3.text(optimal_pf['threshold'], optimal_pf['profit_factor'], 
                f'Optimal: {optimal_pf["threshold"]:.3f}', 
                rotation=90, verticalalignment='bottom')
        
        # 4. Trade Frequency vs Threshold
        ax4 = axes[1, 0]
        ax4.plot(results_df['threshold'], results_df['trade_frequency'], 'c-o', linewidth=2, markersize=6)
        ax4.set_xlabel('Signal Threshold')
        ax4.set_ylabel('Trade Frequency')
        ax4.set_title('Trade Frequency vs Threshold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Cost Drag vs Threshold
        ax5 = axes[1, 1]
        ax5.plot(results_df['threshold'], results_df['cost_drag'], 'r-o', linewidth=2, markersize=6)
        ax5.set_xlabel('Signal Threshold')
        ax5.set_ylabel('Cost Drag (Costs/Gross Returns)')
        ax5.set_title('Cost Drag vs Threshold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Max Drawdown vs Threshold
        ax6 = axes[1, 2]
        ax6.plot(results_df['threshold'], results_df['max_drawdown'], 'orange', marker='o', linewidth=2, markersize=6)
        ax6.set_xlabel('Signal Threshold')
        ax6.set_ylabel('Max Drawdown')
        ax6.set_title('Max Drawdown vs Threshold')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Optimization plots saved to: {save_path}")
    
    def save_optimization_results(self, results_df, save_path="threshold_optimization_results.csv"):
        """Save optimization results to CSV"""
        results_df.to_csv(save_path, index=False)
        print(f"✅ Optimization results saved to: {save_path}")
        
        # Print summary table
        print("\n📊 Threshold Optimization Summary:")
        print("=" * 80)
        print(f"{'Threshold':<10} {'Net Return':<12} {'Sharpe':<8} {'Trades':<8} {'Cost Drag':<10} {'PF':<8}")
        print("-" * 80)
        
        for _, row in results_df.iterrows():
            print(f"{row['threshold']:<10.3f} {row['annual_net_return']:<12.4f} "
                  f"{row['sharpe_ratio']:<8.3f} {row['num_trades']:<8.0f} "
                  f"{row['cost_drag']:<10.3f} {row['profit_factor']:<8.3f}")
        
        return results_df
    
    def run_optimization(self, data_path, output_dir=".", threshold_range=(0.5, 0.95), num_thresholds=20):
        """Run complete threshold optimization"""
        print("🚀 Starting cost-aware threshold optimization...")
        print(f"📊 Transaction cost: {self.transaction_cost:.4f}")
        print(f"📊 Slippage: {self.slippage:.4f}")
        print(f"📊 Total cost per trade: {self.total_cost_per_trade:.4f}")
        print("=" * 60)
        
        # Run optimization
        results = self.optimize_threshold(data_path, threshold_range, num_thresholds)
        
        if results is None:
            print("❌ Optimization failed!")
            return None
        
        results_df, optimal_sharpe, optimal_return, optimal_pf = results
        
        # Create plots
        plot_path = f"{output_dir}/threshold_optimization_results.png"
        self.plot_optimization_results(results_df, plot_path)
        
        # Save results
        results_path = f"{output_dir}/threshold_optimization_results.csv"
        self.save_optimization_results(results_df, results_path)
        
        print(f"\n🎉 Threshold optimization completed successfully!")
        print(f"📁 Results saved to: {results_path}")
        print(f"📊 Plots saved to: {plot_path}")
        
        return results_df, optimal_sharpe, optimal_return, optimal_pf

def main():
    parser = argparse.ArgumentParser(description="Cost-aware signal threshold optimizer")
    parser.add_argument("--data", type=str, default="enhanced_regime_features.csv",
                       help="Path to enhanced dataset")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Output directory for results")
    parser.add_argument("--transaction_cost", type=float, default=0.002,
                       help="Transaction cost as decimal")
    parser.add_argument("--slippage", type=float, default=0.005,
                       help="Slippage per trade as decimal")
    parser.add_argument("--threshold_min", type=float, default=0.5,
                       help="Minimum threshold to test")
    parser.add_argument("--threshold_max", type=float, default=0.95,
                       help="Maximum threshold to test")
    parser.add_argument("--num_thresholds", type=int, default=20,
                       help="Number of thresholds to test")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = CostAwareThresholdOptimizer(
        transaction_cost=args.transaction_cost,
        slippage=args.slippage
    )
    
    # Run optimization
    results = optimizer.run_optimization(
        data_path=args.data,
        output_dir=args.output_dir,
        threshold_range=(args.threshold_min, args.threshold_max),
        num_thresholds=args.num_thresholds
    )
    
    if results is not None:
        results_df, optimal_sharpe, optimal_return, optimal_pf = results
        
        print(f"\n🏆 Recommended Thresholds:")
        print(f"   For Sharpe Ratio: {optimal_sharpe['threshold']:.3f}")
        print(f"   For Net Returns: {optimal_return['threshold']:.3f}")
        print(f"   For Profit Factor: {optimal_pf['threshold']:.3f}")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Use the optimal threshold in your pipeline:")
        print(f"      python run_portfolio_pipeline.py --threshold {optimal_sharpe['threshold']:.3f}")
        print(f"   2. Review the plots to understand the trade-offs")
        print(f"   3. Consider the cost impact on your strategy")

if __name__ == "__main__":
    main() 