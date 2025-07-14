#!/usr/bin/env python3
"""
Markowitz Portfolio Optimizer
Computes optimal portfolio weights using strategy returns across currency pairs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MarkowitzOptimizer:
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize Markowitz optimizer
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        
    def load_returns_data(self, returns_path):
        """Load strategy returns matrix"""
        print("🔄 Loading strategy returns data...")
        
        returns_matrix = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        print(f"✅ Loaded returns data: {returns_matrix.shape}")
        print(f"   Date range: {returns_matrix.index.min()} to {returns_matrix.index.max()}")
        print(f"   Currency pairs: {list(returns_matrix.columns)}")
        
        return returns_matrix
    
    def calculate_portfolio_metrics(self, returns_matrix, weights):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        # Annualize returns (assuming 4-hour data, 6 observations per day)
        annual_factor = 6 * 252  # 6 observations per day * 252 trading days
        
        # Portfolio returns
        portfolio_returns = (returns_matrix * weights).sum(axis=1)
        
        # Annualized metrics
        annual_return = portfolio_returns.mean() * annual_factor
        annual_volatility = portfolio_returns.std() * np.sqrt(annual_factor)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_returns': portfolio_returns
        }
    
    def portfolio_variance(self, weights, returns_matrix):
        """Calculate portfolio variance (for minimization)"""
        portfolio_returns = (returns_matrix * weights).sum(axis=1)
        return portfolio_returns.var()
    
    def negative_sharpe_ratio(self, weights, returns_matrix):
        """Calculate negative Sharpe ratio (for maximization)"""
        metrics = self.calculate_portfolio_metrics(returns_matrix, weights)
        return -metrics['sharpe_ratio']
    
    def optimize_minimum_variance(self, returns_matrix):
        """Find minimum variance portfolio"""
        print("🔄 Optimizing minimum variance portfolio...")
        
        n_assets = len(returns_matrix.columns)
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        )
        bounds = tuple((0, 1) for _ in range(n_assets))  # weights between 0 and 1
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            self.portfolio_variance,
            initial_weights,
            args=(returns_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            min_var_weights = result.x
            min_var_metrics = self.calculate_portfolio_metrics(returns_matrix, min_var_weights)
            
            print("✅ Minimum variance portfolio optimized")
            print(f"   Annual Return: {min_var_metrics['annual_return']:.4f}")
            print(f"   Annual Volatility: {min_var_metrics['annual_volatility']:.4f}")
            print(f"   Sharpe Ratio: {min_var_metrics['sharpe_ratio']:.4f}")
            
            return min_var_weights, min_var_metrics
        else:
            print(f"❌ Minimum variance optimization failed: {result.message}")
            return None, None
    
    def optimize_maximum_sharpe(self, returns_matrix):
        """Find maximum Sharpe ratio portfolio"""
        print("🔄 Optimizing maximum Sharpe ratio portfolio...")
        
        n_assets = len(returns_matrix.columns)
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        )
        bounds = tuple((0, 1) for _ in range(n_assets))  # weights between 0 and 1
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            self.negative_sharpe_ratio,
            initial_weights,
            args=(returns_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            max_sharpe_weights = result.x
            max_sharpe_metrics = self.calculate_portfolio_metrics(returns_matrix, max_sharpe_weights)
            
            print("✅ Maximum Sharpe ratio portfolio optimized")
            print(f"   Annual Return: {max_sharpe_metrics['annual_return']:.4f}")
            print(f"   Annual Volatility: {max_sharpe_metrics['annual_volatility']:.4f}")
            print(f"   Sharpe Ratio: {max_sharpe_metrics['sharpe_ratio']:.4f}")
            
            return max_sharpe_weights, max_sharpe_metrics
        else:
            print(f"❌ Maximum Sharpe optimization failed: {result.message}")
            return None, None
    
    def generate_efficient_frontier(self, returns_matrix, n_points=50):
        """Generate efficient frontier"""
        print("🔄 Generating efficient frontier...")
        
        # Get minimum variance portfolio
        min_var_weights, min_var_metrics = self.optimize_minimum_variance(returns_matrix)
        if min_var_weights is None:
            return None
        
        # Get maximum Sharpe portfolio
        max_sharpe_weights, max_sharpe_metrics = self.optimize_maximum_sharpe(returns_matrix)
        if max_sharpe_weights is None:
            return None
        
        # Generate target returns between min and max
        min_return = min_var_metrics['annual_return']
        max_return = max_sharpe_metrics['annual_return']
        target_returns = np.linspace(min_return, max_return * 1.2, n_points)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            n_assets = len(returns_matrix.columns)
            
            # Constraints: weights sum to 1, target return, all weights >= 0
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
                {'type': 'eq', 'fun': lambda x: self.calculate_portfolio_metrics(returns_matrix, x)['annual_return'] - target_return},  # target return
            )
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess: equal weights
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                self.portfolio_variance,
                initial_weights,
                args=(returns_matrix,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                metrics = self.calculate_portfolio_metrics(returns_matrix, weights)
                efficient_portfolios.append({
                    'weights': weights,
                    'return': metrics['annual_return'],
                    'volatility': metrics['annual_volatility'],
                    'sharpe': metrics['sharpe_ratio']
                })
        
        print(f"✅ Generated {len(efficient_portfolios)} efficient frontier points")
        return efficient_portfolios
    
    def plot_optimization_results(self, returns_matrix, min_var_weights, max_sharpe_weights, 
                                efficient_portfolios, save_path="markowitz_optimization.png"):
        """Plot optimization results"""
        print("🔄 Creating optimization plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Efficient Frontier
        if efficient_portfolios:
            returns = [p['return'] for p in efficient_portfolios]
            volatilities = [p['volatility'] for p in efficient_portfolios]
            sharpes = [p['sharpe'] for p in efficient_portfolios]
            
            ax1 = axes[0, 0]
            scatter = ax1.scatter(volatilities, returns, c=sharpes, cmap='viridis', s=50, alpha=0.7)
            ax1.set_xlabel('Annual Volatility')
            ax1.set_ylabel('Annual Return')
            ax1.set_title('Efficient Frontier')
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Sharpe Ratio')
            
            # Mark special portfolios
            min_var_metrics = self.calculate_portfolio_metrics(returns_matrix, min_var_weights)
            max_sharpe_metrics = self.calculate_portfolio_metrics(returns_matrix, max_sharpe_weights)
            
            ax1.scatter(min_var_metrics['annual_volatility'], min_var_metrics['annual_return'], 
                       color='red', s=100, marker='*', label='Min Variance', zorder=5)
            ax1.scatter(max_sharpe_metrics['annual_volatility'], max_sharpe_metrics['annual_return'], 
                       color='green', s=100, marker='*', label='Max Sharpe', zorder=5)
            ax1.legend()
        
        # 2. Portfolio Weights Comparison
        ax2 = axes[0, 1]
        x = np.arange(len(returns_matrix.columns))
        width = 0.35
        
        ax2.bar(x - width/2, min_var_weights, width, label='Min Variance', alpha=0.8)
        ax2.bar(x + width/2, max_sharpe_weights, width, label='Max Sharpe', alpha=0.8)
        ax2.set_xlabel('Currency Pairs')
        ax2.set_ylabel('Weight')
        ax2.set_title('Portfolio Weights Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(returns_matrix.columns, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Individual Asset Performance
        ax3 = axes[1, 0]
        annual_factor = 6 * 252
        asset_returns = returns_matrix.mean() * annual_factor
        asset_vols = returns_matrix.std() * np.sqrt(annual_factor)
        asset_sharpes = asset_returns / asset_vols
        
        ax3.scatter(asset_vols, asset_returns, s=100, alpha=0.7)
        for i, pair in enumerate(returns_matrix.columns):
            ax3.annotate(pair, (asset_vols[i], asset_returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('Annual Volatility')
        ax3.set_ylabel('Annual Return')
        ax3.set_title('Individual Asset Performance')
        ax3.grid(True, alpha=0.3)
        
        # 4. Correlation Heatmap
        ax4 = axes[1, 1]
        correlation_matrix = returns_matrix.corr()
        im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(returns_matrix.columns)))
        ax4.set_yticks(range(len(returns_matrix.columns)))
        ax4.set_xticklabels(returns_matrix.columns, rotation=45)
        ax4.set_yticklabels(returns_matrix.columns)
        ax4.set_title('Returns Correlation Matrix')
        
        # Add correlation values
        for i in range(len(returns_matrix.columns)):
            for j in range(len(returns_matrix.columns)):
                text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Optimization plots saved to: {save_path}")
    
    def save_optimization_results(self, returns_matrix, min_var_weights, max_sharpe_weights, 
                                efficient_portfolios, save_path="markowitz_results.csv"):
        """Save optimization results to CSV"""
        print("🔄 Saving optimization results...")
        
        results = []
        
        # Individual assets
        annual_factor = 6 * 252
        for i, pair in enumerate(returns_matrix.columns):
            asset_returns = returns_matrix[pair].mean() * annual_factor
            asset_vols = returns_matrix[pair].std() * np.sqrt(annual_factor)
            asset_sharpe = asset_returns / asset_vols if asset_vols > 0 else 0
            
            results.append({
                'Asset': pair,
                'Type': 'Individual',
                'Weight': 1.0,
                'Annual_Return': asset_returns,
                'Annual_Volatility': asset_vols,
                'Sharpe_Ratio': asset_sharpe
            })
        
        # Minimum variance portfolio
        min_var_metrics = self.calculate_portfolio_metrics(returns_matrix, min_var_weights)
        for i, pair in enumerate(returns_matrix.columns):
            results.append({
                'Asset': pair,
                'Type': 'Min_Variance',
                'Weight': min_var_weights[i],
                'Annual_Return': min_var_metrics['annual_return'],
                'Annual_Volatility': min_var_metrics['annual_volatility'],
                'Sharpe_Ratio': min_var_metrics['sharpe_ratio']
            })
        
        # Maximum Sharpe portfolio
        max_sharpe_metrics = self.calculate_portfolio_metrics(returns_matrix, max_sharpe_weights)
        for i, pair in enumerate(returns_matrix.columns):
            results.append({
                'Asset': pair,
                'Type': 'Max_Sharpe',
                'Weight': max_sharpe_weights[i],
                'Annual_Return': max_sharpe_metrics['annual_return'],
                'Annual_Volatility': max_sharpe_metrics['annual_volatility'],
                'Sharpe_Ratio': max_sharpe_metrics['sharpe_ratio']
            })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path, index=False)
        print(f"✅ Optimization results saved to: {save_path}")
        
        return results_df
    
    def run_optimization(self, returns_path, output_dir="."):
        """Run complete Markowitz optimization"""
        print("🚀 Starting Markowitz portfolio optimization...")
        
        # Load data
        returns_matrix = self.load_returns_data(returns_path)
        
        # Optimize portfolios
        min_var_weights, min_var_metrics = self.optimize_minimum_variance(returns_matrix)
        max_sharpe_weights, max_sharpe_metrics = self.optimize_maximum_sharpe(returns_matrix)
        
        if min_var_weights is None or max_sharpe_weights is None:
            print("❌ Optimization failed!")
            return None
        
        # Generate efficient frontier
        efficient_portfolios = self.generate_efficient_frontier(returns_matrix)
        
        # Create plots
        plot_path = f"{output_dir}/markowitz_optimization.png"
        self.plot_optimization_results(returns_matrix, min_var_weights, max_sharpe_weights, 
                                     efficient_portfolios, plot_path)
        
        # Save results
        results_path = f"{output_dir}/markowitz_results.csv"
        results_df = self.save_optimization_results(returns_matrix, min_var_weights, max_sharpe_weights, 
                                                  efficient_portfolios, results_path)
        
        print("\n🎉 Markowitz optimization completed successfully!")
        print(f"📁 Results saved to: {results_path}")
        print(f"📊 Plots saved to: {plot_path}")
        
        return {
            'min_var_weights': min_var_weights,
            'max_sharpe_weights': max_sharpe_weights,
            'efficient_portfolios': efficient_portfolios,
            'results_df': results_df
        }

def main():
    parser = argparse.ArgumentParser(description="Markowitz portfolio optimization")
    parser.add_argument("--returns", type=str, default="strategy_returns.csv",
                       help="Path to strategy returns matrix")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Output directory for results")
    parser.add_argument("--risk_free_rate", type=float, default=0.02,
                       help="Annual risk-free rate")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = MarkowitzOptimizer(risk_free_rate=args.risk_free_rate)
    
    # Run optimization
    results = optimizer.run_optimization(
        returns_path=args.returns,
        output_dir=args.output_dir
    )
    
    if results is not None:
        print("\n📈 Optimization Summary:")
        print("Minimum Variance Portfolio:")
        print(f"  Weights: {dict(zip(['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'NZDUSD'], results['min_var_weights']))}")
        print("Maximum Sharpe Portfolio:")
        print(f"  Weights: {dict(zip(['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'NZDUSD'], results['max_sharpe_weights']))}")

if __name__ == "__main__":
    main() 