#!/usr/bin/env python3
"""
Head-to-Head Backtest Comparison
Compares models with and without bull_prob_hmm using different thresholds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BacktestEngine:
    def __init__(self, transaction_cost=0.0, rolling_window=200):
        """
        Initialize backtest engine
        
        Args:
            transaction_cost: Fixed transaction cost as decimal (e.g., 0.0005 = 5 bps)
            rolling_window: Window size for rolling metrics
        """
        self.transaction_cost = transaction_cost
        self.rolling_window = rolling_window
        
    def load_model_and_data(self, model_path, threshold_path, data_path):
        """Load model, threshold info, and test data"""
        # Load model
        model = joblib.load(model_path)
        
        # Load threshold info
        with open(threshold_path, 'r') as f:
            threshold_info = json.load(f)
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Create label column if it doesn't exist
        if 'label' not in data.columns and 'return' in data.columns:
            # Create label: positive return = 1, negative/zero = 0
            data['label'] = (data['return'] > 0).astype(int)
        
        # Prepare features and target
        feature_names = threshold_info['feature_names']
        X = data[feature_names]
        y = data['label'] if 'label' in data.columns else None
        
        # Get predictions
        y_proba = model.predict_proba(X)[:, 1]
        
        return model, threshold_info, X, y, y_proba
    
    def generate_signals(self, y_proba, threshold):
        """Generate trading signals (1 = long, 0 = flat)"""
        return (y_proba > threshold).astype(int)
    
    def calculate_returns(self, data, signals):
        """Calculate strategy returns"""
        # Get price returns (assuming 'return' column exists)
        if 'return' in data.columns:
            price_returns = data['return']
        elif 'close' in data.columns:
            price_returns = data['close'].pct_change()
        else:
            raise ValueError("Need 'return' or 'close' column in data")
        
        # Strategy returns: long when signal=1, flat when signal=0
        strategy_returns = signals * price_returns
        
        # Apply transaction costs
        if self.transaction_cost > 0:
            # Cost when entering position (signal changes from 0 to 1)
            entry_costs = (signals.diff() == 1) * self.transaction_cost
            strategy_returns -= entry_costs
        
        return strategy_returns
    
    def calculate_metrics(self, y_true, signals, returns):
        """Calculate performance metrics"""
        # Basic classification metrics
        precision = precision_score(y_true, signals, zero_division=0)
        recall = recall_score(y_true, signals, zero_division=0)
        f1 = f1_score(y_true, signals, zero_division=0)
        accuracy = accuracy_score(y_true, signals)
        
        # Trading metrics
        total_return = returns.sum()
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        max_drawdown = self.calculate_max_drawdown(returns)
        win_rate = (returns > 0).mean()
        
        # Trade frequency
        trade_frequency = signals.mean()
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trade_frequency': trade_frequency
        }
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_rolling_metrics(self, y_true, signals, returns, window=None):
        """Calculate rolling metrics"""
        if window is None:
            window = self.rolling_window
        
        # Ensure y_true and signals are pandas Series
        if isinstance(y_true, np.ndarray):
            y_true = pd.Series(y_true)
        if isinstance(signals, np.ndarray):
            signals = pd.Series(signals)
        
        # Rolling precision and recall
        rolling_precision = []
        rolling_recall = []
        rolling_f1 = []
        
        for i in range(window, len(y_true)):
            y_window = y_true.iloc[i-window:i]
            s_window = signals.iloc[i-window:i]
            
            if s_window.sum() > 0:  # Only if there are positive predictions
                p = precision_score(y_window, s_window, zero_division=0)
                r = recall_score(y_window, s_window, zero_division=0)
                f = f1_score(y_window, s_window, zero_division=0)
            else:
                p = r = f = 0
            
            rolling_precision.append(p)
            rolling_recall.append(r)
            rolling_f1.append(f)
        
        # Rolling returns
        rolling_returns = returns.rolling(window=window).sum()
        
        # Rolling trade frequency
        rolling_trade_freq = signals.rolling(window=window).mean()
        
        return {
            'precision': rolling_precision,
            'recall': rolling_recall,
            'f1': rolling_f1,
            'returns': rolling_returns,
            'trade_frequency': rolling_trade_freq
        }
    
    def run_backtest(self, model_path, threshold_path, data_path, threshold, model_name):
        """Run complete backtest for one model/threshold combination"""
        print(f"🔄 Running backtest for {model_name} (threshold={threshold})")
        
        # Load model and data
        model, threshold_info, X, y, y_proba = self.load_model_and_data(
            model_path, threshold_path, data_path
        )
        
        # Generate signals
        signals = self.generate_signals(y_proba, threshold)
        
        # Calculate returns
        data = pd.read_csv(data_path)
        returns = self.calculate_returns(data, signals)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, signals, returns)
        
        # Calculate rolling metrics
        rolling_metrics = self.calculate_rolling_metrics(y, signals, returns)
        
        # Create results dictionary
        results = {
            'model_name': model_name,
            'threshold': threshold,
            'signals': signals,
            'returns': returns,
            'y_true': y,
            'y_proba': y_proba,
            'metrics': metrics,
            'rolling_metrics': rolling_metrics
        }
        
        print(f"✅ Completed backtest for {model_name}")
        print(f"   Total Return: {metrics['total_return']:.4f}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"   Trade Frequency: {metrics['trade_frequency']:.3f}")
        
        return results
    
    def plot_comparison(self, results_list, save_path="backtest_comparison.png"):
        """Create comprehensive comparison plots"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison: With vs Without bull_prob_hmm', fontsize=16, fontweight='bold')
        
        # Colors for different models
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, results in enumerate(results_list):
            color = colors[i % len(colors)]
            label = f"{results['model_name']} (t={results['threshold']})"
            
            # 1. Cumulative Returns
            cumulative_returns = (1 + results['returns']).cumprod()
            axes[0, 0].plot(cumulative_returns.index, cumulative_returns, 
                           label=label, color=color, linewidth=2)
            
            # 2. Drawdown
            cumulative = (1 + results['returns']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            axes[0, 1].fill_between(drawdown.index, drawdown, 0, 
                                   alpha=0.3, color=color, label=label)
            axes[0, 1].plot(drawdown.index, drawdown, color=color, linewidth=1)
            
            # 3. Rolling Precision
            if len(results['rolling_metrics']['precision']) > 0:
                x_vals = range(len(results['rolling_metrics']['precision']))
                axes[1, 0].plot(x_vals, results['rolling_metrics']['precision'], 
                               label=label, color=color, linewidth=1, alpha=0.7)
            
            # 4. Rolling Recall
            if len(results['rolling_metrics']['recall']) > 0:
                x_vals = range(len(results['rolling_metrics']['recall']))
                axes[1, 1].plot(x_vals, results['rolling_metrics']['recall'], 
                               label=label, color=color, linewidth=1, alpha=0.7)
            
            # 5. Trade Frequency
            if len(results['rolling_metrics']['trade_frequency']) > 0:
                axes[2, 0].plot(results['rolling_metrics']['trade_frequency'].index, 
                               results['rolling_metrics']['trade_frequency'], 
                               label=label, color=color, linewidth=1, alpha=0.7)
            
            # 6. Rolling Returns
            if len(results['rolling_metrics']['returns']) > 0:
                axes[2, 1].plot(results['rolling_metrics']['returns'].index, 
                               results['rolling_metrics']['returns'], 
                               label=label, color=color, linewidth=1, alpha=0.7)
        
        # Set labels and titles
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Rolling Precision')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Rolling Recall')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[2, 0].set_title('Rolling Trade Frequency')
        axes[2, 0].set_ylabel('Trade Frequency')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].set_title('Rolling Returns')
        axes[2, 1].set_ylabel('Returns')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Comparison plots saved as {save_path}")
    
    def create_summary_table(self, results_list, save_path="backtest_summary.csv"):
        """Create summary table of all results"""
        summary_data = []
        
        for results in results_list:
            metrics = results['metrics']
            summary_data.append({
                'Model': results['model_name'],
                'Threshold': results['threshold'],
                'Total_Return': metrics['total_return'],
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Max_Drawdown': metrics['max_drawdown'],
                'Win_Rate': metrics['win_rate'],
                'Trade_Frequency': metrics['trade_frequency'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1'],
                'Accuracy': metrics['accuracy']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(save_path, index=False)
        print(f"✅ Summary table saved as {save_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False, float_format='%.4f'))
        print("="*80)
        
        return summary_df

def main():
    parser = argparse.ArgumentParser(description="Run head-to-head backtest comparison")
    parser.add_argument("--data", type=str, default="merged_with_regime_features.csv", 
                       help="Path to data file")
    parser.add_argument("--rolling_window", type=int, default=200, 
                       help="Rolling window size for metrics")
    parser.add_argument("--transaction_cost", type=float, default=0.0, 
                       help="Transaction cost as decimal (e.g., 0.0005 = 5 bps)")
    args = parser.parse_args()
    
    # Initialize backtest engine
    engine = BacktestEngine(
        transaction_cost=args.transaction_cost,
        rolling_window=args.rolling_window
    )
    
    # Define model configurations
    model_configs = [
        {
            'model_path': 'corrective_ai_lgbm.pkl',
            'threshold_path': 'corrective_ai_lgbm_threshold.json',
            'model_name': 'With bull_prob_hmm',
            'thresholds': [0.05, 0.20]
        },
        {
            'model_path': 'corrective_ai_lgbm_nobull.pkl',
            'threshold_path': 'corrective_ai_lgbm_nobull_threshold.json',
            'model_name': 'Without bull_prob_hmm',
            'thresholds': [0.05, 0.20]
        }
    ]
    
    # Run backtests
    all_results = []
    
    for config in model_configs:
        for threshold in config['thresholds']:
            try:
                results = engine.run_backtest(
                    model_path=config['model_path'],
                    threshold_path=config['threshold_path'],
                    data_path=args.data,
                    threshold=threshold,
                    model_name=f"{config['model_name']} (t={threshold})"
                )
                all_results.append(results)
            except Exception as e:
                print(f"❌ Error running backtest for {config['model_name']} (t={threshold}): {e}")
    
    if not all_results:
        print("❌ No successful backtests completed")
        return
    
    # Create comparison plots
    engine.plot_comparison(all_results)
    
    # Create summary table
    engine.create_summary_table(all_results)
    
    print("\n🎉 Backtest comparison completed successfully!")

if __name__ == "__main__":
    main() 