#!/usr/bin/env python3
"""
Portfolio Backtest Engine
Backtests optimized portfolios against individual currencies and equal-weight benchmark.
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

class PortfolioBacktest:
    def __init__(self, transaction_cost=0.0, slippage=0.0, rebalance_frequency='daily'):
        """
        Initialize portfolio backtest engine
        
        Args:
            transaction_cost: Fixed transaction cost as decimal
            slippage: Slippage per trade as decimal
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.rebalance_frequency = rebalance_frequency
        
    def load_data(self, returns_path, weights_path):
        """Load strategy returns and portfolio weights"""
        print("🔄 Loading backtest data...")
        
        # Load strategy returns
        returns_matrix = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        # Load portfolio weights
        weights_df = pd.read_csv(weights_path)
        
        print(f"✅ Loaded returns data: {returns_matrix.shape}")
        print(f"✅ Loaded weights data: {weights_df.shape}")
        
        return returns_matrix, weights_df
    
    def extract_portfolio_weights(self, weights_df, portfolio_type):
        """Extract weights for specific portfolio type"""
        portfolio_data = weights_df[weights_df['Type'] == portfolio_type]
        
        if len(portfolio_data) == 0:
            print(f"⚠️ No weights found for {portfolio_type}")
            return None
        
        weights = {}
        for _, row in portfolio_data.iterrows():
            weights[row['Asset']] = row['Weight']
        
        return weights
    
    def calculate_portfolio_returns(self, returns_matrix, weights, rebalance_freq='daily'):
        """Calculate portfolio returns with rebalancing"""
        print(f"🔄 Calculating {rebalance_freq} portfolio returns...")
        
        # Convert weights dict to array in same order as returns_matrix columns
        weight_array = np.array([weights.get(col, 0) for col in returns_matrix.columns])
        
        # Calculate portfolio returns
        portfolio_returns = (returns_matrix * weight_array).sum(axis=1)
        
        # Apply transaction costs and slippage on rebalancing (assume rebalancing at each period)
        if self.transaction_cost > 0 or self.slippage > 0:
            # Assume a trade occurs whenever weights are nonzero (conservative)
            trade_cost = (self.transaction_cost + self.slippage) * np.sum(weight_array > 0)
            portfolio_returns -= trade_cost
        
        # Apply rebalancing if needed
        if rebalance_freq != 'daily':
            # Resample to rebalancing frequency
            if rebalance_freq == 'weekly':
                portfolio_returns = portfolio_returns.resample('W').sum()
            elif rebalance_freq == 'monthly':
                portfolio_returns = portfolio_returns.resample('M').sum()
        
        return portfolio_returns
    
    def calculate_equal_weight_returns(self, returns_matrix, rebalance_freq='daily'):
        """Calculate equal-weight portfolio returns"""
        n_assets = len(returns_matrix.columns)
        equal_weights = {col: 1/n_assets for col in returns_matrix.columns}
        
        return self.calculate_portfolio_returns(returns_matrix, equal_weights, rebalance_freq)
    
    def calculate_performance_metrics(self, returns, name="Portfolio"):
        """Calculate comprehensive performance metrics"""
        # Annualize returns (assuming 4-hour data, 6 observations per day)
        annual_factor = 6 * 252
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * annual_factor
        annual_volatility = returns.std() * np.sqrt(annual_factor)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(returns)
        var_95 = np.percentile(returns, 5)  # 95% VaR
        cvar_95 = returns[returns <= var_95].mean()  # Conditional VaR
        
        # Win rate and profit factor
        win_rate = (returns > 0).mean()
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calmar ratio (annual return / max drawdown)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'name': name,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio
        }
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def run_portfolio_backtest(self, returns_matrix, weights_df):
        """Run comprehensive portfolio backtest"""
        print("🚀 Starting portfolio backtest...")
        
        # Extract portfolio weights
        min_var_weights = self.extract_portfolio_weights(weights_df, 'Min_Variance')
        max_sharpe_weights = self.extract_portfolio_weights(weights_df, 'Max_Sharpe')
        
        if min_var_weights is None or max_sharpe_weights is None:
            print("❌ Could not extract portfolio weights!")
            return None
        
        # Calculate portfolio returns
        min_var_returns = self.calculate_portfolio_returns(returns_matrix, min_var_weights, self.rebalance_frequency)
        max_sharpe_returns = self.calculate_portfolio_returns(returns_matrix, max_sharpe_weights, self.rebalance_frequency)
        equal_weight_returns = self.calculate_equal_weight_returns(returns_matrix, self.rebalance_frequency)
        
        # Calculate individual asset returns
        individual_returns = {}
        annual_factor = 6 * 252
        for col in returns_matrix.columns:
            individual_returns[col] = returns_matrix[col] * annual_factor
        
        # Calculate performance metrics
        results = []
        
        # Individual assets
        for pair, returns in individual_returns.items():
            metrics = self.calculate_performance_metrics(returns, pair)
            results.append(metrics)
        
        # Portfolios
        min_var_metrics = self.calculate_performance_metrics(min_var_returns, "Min_Variance")
        max_sharpe_metrics = self.calculate_performance_metrics(max_sharpe_returns, "Max_Sharpe")
        equal_weight_metrics = self.calculate_performance_metrics(equal_weight_returns, "Equal_Weight")
        
        results.extend([min_var_metrics, max_sharpe_metrics, equal_weight_metrics])
        
        return {
            'results': results,
            'returns': {
                'min_var': min_var_returns,
                'max_sharpe': max_sharpe_returns,
                'equal_weight': equal_weight_returns,
                'individual': individual_returns
            }
        }
    
    def plot_backtest_results(self, backtest_results, save_path="portfolio_backtest_results.png"):
        """Plot comprehensive backtest results"""
        print("🔄 Creating backtest plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        
        # Individual assets
        for pair, returns in backtest_results['returns']['individual'].items():
            cumulative = (1 + returns).cumprod()
            ax1.plot(cumulative.index, cumulative, label=pair, alpha=0.7, linewidth=1)
        
        # Portfolios
        min_var_cum = (1 + backtest_results['returns']['min_var']).cumprod()
        max_sharpe_cum = (1 + backtest_results['returns']['max_sharpe']).cumprod()
        equal_weight_cum = (1 + backtest_results['returns']['equal_weight']).cumprod()
        
        ax1.plot(min_var_cum.index, min_var_cum, label='Min Variance', linewidth=2, color='red')
        ax1.plot(max_sharpe_cum.index, max_sharpe_cum, label='Max Sharpe', linewidth=2, color='green')
        ax1.plot(equal_weight_cum.index, equal_weight_cum, label='Equal Weight', linewidth=2, color='blue')
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title('Cumulative Returns Comparison')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Metrics Heatmap
        ax2 = axes[0, 1]
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(backtest_results['results'])
        metrics_df = metrics_df.set_index('name')
        
        # Select key metrics for heatmap
        heatmap_metrics = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
        heatmap_data = metrics_df[heatmap_metrics].T
        
        im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        ax2.set_xticks(range(len(heatmap_data.columns)))
        ax2.set_yticks(range(len(heatmap_data.index)))
        ax2.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
        ax2.set_yticklabels(heatmap_data.index)
        ax2.set_title('Performance Metrics Heatmap')
        
        # Add values to heatmap
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                if not np.isnan(value):
                    text = ax2.text(j, i, f'{value:.3f}', ha="center", va="center", 
                                   color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax2)
        
        # 3. Rolling Sharpe Ratio
        ax3 = axes[1, 0]
        
        # Calculate rolling Sharpe for portfolios
        window = 252  # 1 year
        min_var_rolling_sharpe = backtest_results['returns']['min_var'].rolling(window).mean() / backtest_results['returns']['min_var'].rolling(window).std()
        max_sharpe_rolling_sharpe = backtest_results['returns']['max_sharpe'].rolling(window).mean() / backtest_results['returns']['max_sharpe'].rolling(window).std()
        equal_weight_rolling_sharpe = backtest_results['returns']['equal_weight'].rolling(window).mean() / backtest_results['returns']['equal_weight'].rolling(window).std()
        
        ax3.plot(min_var_rolling_sharpe.index, min_var_rolling_sharpe, label='Min Variance', alpha=0.8)
        ax3.plot(max_sharpe_rolling_sharpe.index, max_sharpe_rolling_sharpe, label='Max Sharpe', alpha=0.8)
        ax3.plot(equal_weight_rolling_sharpe.index, equal_weight_rolling_sharpe, label='Equal Weight', alpha=0.8)
        
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Rolling Sharpe Ratio')
        ax3.set_title(f'Rolling Sharpe Ratio ({window}-day window)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Drawdown Analysis
        ax4 = axes[1, 1]
        
        # Calculate drawdowns
        min_var_dd = self.calculate_drawdown_series(backtest_results['returns']['min_var'])
        max_sharpe_dd = self.calculate_drawdown_series(backtest_results['returns']['max_sharpe'])
        equal_weight_dd = self.calculate_drawdown_series(backtest_results['returns']['equal_weight'])
        
        ax4.fill_between(min_var_dd.index, min_var_dd, 0, alpha=0.3, label='Min Variance')
        ax4.fill_between(max_sharpe_dd.index, max_sharpe_dd, 0, alpha=0.3, label='Max Sharpe')
        ax4.fill_between(equal_weight_dd.index, equal_weight_dd, 0, alpha=0.3, label='Equal Weight')
        
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Drawdown')
        ax4.set_title('Portfolio Drawdowns')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Backtest plots saved to: {save_path}")
    
    def calculate_drawdown_series(self, returns):
        """Calculate drawdown series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    def save_backtest_results(self, backtest_results, save_path="portfolio_backtest_summary.csv"):
        """Save backtest results to CSV"""
        print("🔄 Saving backtest results...")
        
        results_df = pd.DataFrame(backtest_results['results'])
        results_df.to_csv(save_path, index=False)
        
        print(f"✅ Backtest results saved to: {save_path}")
        
        # Print summary
        print("\n📊 Portfolio Backtest Summary:")
        print("=" * 80)
        
        # Sort by Sharpe ratio
        results_df_sorted = results_df.sort_values('sharpe_ratio', ascending=False)
        
        for _, row in results_df_sorted.iterrows():
            print(f"{row['name']:15} | "
                  f"Return: {row['annual_return']:8.4f} | "
                  f"Vol: {row['annual_volatility']:8.4f} | "
                  f"Sharpe: {row['sharpe_ratio']:6.3f} | "
                  f"DD: {row['max_drawdown']:6.3f} | "
                  f"PF: {row['profit_factor']:6.2f}")
        
        return results_df
    
    def run_complete_backtest(self, returns_path, weights_path, output_dir="."):
        """Run complete portfolio backtest"""
        print("🚀 Starting complete portfolio backtest...")
        
        # Load data
        returns_matrix, weights_df = self.load_data(returns_path, weights_path)
        
        # Run backtest
        backtest_results = self.run_portfolio_backtest(returns_matrix, weights_df)
        
        if backtest_results is None:
            print("❌ Backtest failed!")
            return None
        
        # Create plots
        plot_path = f"{output_dir}/portfolio_backtest_results.png"
        self.plot_backtest_results(backtest_results, plot_path)
        
        # Save results
        results_path = f"{output_dir}/portfolio_backtest_summary.csv"
        results_df = self.save_backtest_results(backtest_results, results_path)
        
        print("\n🎉 Portfolio backtest completed successfully!")
        print(f"📁 Results saved to: {results_path}")
        print(f"📊 Plots saved to: {plot_path}")
        
        return backtest_results

def main():
    parser = argparse.ArgumentParser(description="Portfolio backtest engine")
    parser.add_argument("--returns", type=str, default="strategy_returns.csv",
                       help="Path to strategy returns matrix")
    parser.add_argument("--weights", type=str, default="markowitz_results.csv",
                       help="Path to portfolio weights")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Output directory for results")
    parser.add_argument("--rebalance_freq", type=str, default="daily",
                       choices=['daily', 'weekly', 'monthly'],
                       help="Portfolio rebalancing frequency")
    parser.add_argument("--transaction_cost", type=float, default=0.0,
                       help="Transaction cost as decimal")
    parser.add_argument("--slippage", type=float, default=0.002,
                       help="Slippage per trade as decimal (e.g., 0.002 = 0.2%)")
    
    args = parser.parse_args()
    
    # Initialize backtest engine
    backtest = PortfolioBacktest(
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        rebalance_frequency=args.rebalance_freq
    )
    
    # Run backtest
    results = backtest.run_complete_backtest(
        returns_path=args.returns,
        weights_path=args.weights,
        output_dir=args.output_dir
    )
    
    if results is not None:
        print("\n🏆 Best Performing Portfolio:")
        results_df = pd.DataFrame(results['results'])
        best_portfolio = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        print(f"   {best_portfolio['name']}: Sharpe = {best_portfolio['sharpe_ratio']:.3f}, "
              f"Return = {best_portfolio['annual_return']:.4f}")
        print(f"   Transaction cost: {args.transaction_cost}")
        print(f"   Slippage: {args.slippage}")

if __name__ == "__main__":
    main() 