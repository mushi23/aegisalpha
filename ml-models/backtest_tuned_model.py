#!/usr/bin/env python3
"""
Advanced Backtest for Tuned LightGBM Model
Comprehensive trading simulation with the 74% F1 tuned model.
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

class TunedModelBacktest:
    def __init__(self, transaction_cost=0.0, rolling_window=200):
        """
        Initialize backtest engine
        
        Args:
            transaction_cost: Fixed transaction cost as decimal (e.g., 0.0005 = 5 bps)
            rolling_window: Window size for rolling metrics
        """
        self.transaction_cost = transaction_cost
        self.rolling_window = rolling_window
        
    def load_tuned_model_and_data(self, data_path):
        """Load tuned model and prepare data"""
        print("🔄 Loading tuned model and data...")
        
        # Load tuned model
        try:
            tuned_model = joblib.load('lgbm_best_model.pkl')
            print("✅ Tuned model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading tuned model: {e}")
            return None, None, None, None
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Create label if needed
        if 'label' not in data.columns and 'return' in data.columns:
            cost_per_trade = 0.002 + 0.005
            data['label'] = ((data['return'] - cost_per_trade) > 0).astype(int)
        
        # Load feature list
        with open('feature_list_full_technical.txt', 'r') as f:
            features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        features = [f for f in features if f in data.columns]
        
        X = data[features]
        y = data['label']
        
        # Train-test split (same as training)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, stratify=None
        )
        
        print(f"✅ Data prepared: {len(features)} features, {len(X_test)} test samples")
        
        return tuned_model, X_test, y_test, data.iloc[X_test.index]
    
    def generate_signals(self, model, X_test, threshold=0.5):
        """Generate trading signals (1 = long, 0 = flat)"""
        y_proba = model.predict_proba(X_test)[:, 1]
        return (y_proba > threshold).astype(int), y_proba
    
    def calculate_returns(self, data, signals):
        """Calculate strategy returns"""
        # Get price returns
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
    
    def calculate_trading_metrics(self, y_true, signals, returns, y_proba):
        """Calculate comprehensive trading metrics"""
        # Basic classification metrics
        precision = precision_score(y_true, signals, zero_division=0)
        recall = recall_score(y_true, signals, zero_division=0)
        f1 = f1_score(y_true, signals, zero_division=0)
        accuracy = accuracy_score(y_true, signals)
        
        # Trading metrics
        total_return = returns.sum()
        cumulative_return = (1 + returns).prod() - 1
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        max_drawdown = self.calculate_max_drawdown(returns)
        win_rate = (returns > 0).mean()
        
        # Risk metrics
        volatility = returns.std()
        var_95 = np.percentile(returns, 5)  # 95% VaR
        cvar_95 = returns[returns <= var_95].mean()  # Conditional VaR
        
        # Trade metrics
        trade_frequency = signals.mean()
        avg_trade_return = returns[signals == 1].mean() if (signals == 1).sum() > 0 else 0
        avg_win = returns[(returns > 0) & (signals == 1)].mean() if ((returns > 0) & (signals == 1)).sum() > 0 else 0
        avg_loss = returns[(returns < 0) & (signals == 1)].mean() if ((returns < 0) & (signals == 1)).sum() > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Model confidence metrics
        avg_confidence = y_proba.mean()
        confidence_std = y_proba.std()
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'total_return': total_return,
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'trade_frequency': trade_frequency,
            'avg_trade_return': avg_trade_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std
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
        
        # Ensure inputs are pandas Series
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
            
            if s_window.sum() > 0:
                p = precision_score(y_window, s_window, zero_division=0)
                r = recall_score(y_window, s_window, zero_division=0)
                f = f1_score(y_window, s_window, zero_division=0)
            else:
                p = r = f = 0
            
            rolling_precision.append(p)
            rolling_recall.append(r)
            rolling_f1.append(f)
        
        # Rolling returns and trade frequency
        rolling_returns = returns.rolling(window=window).sum()
        rolling_trade_freq = signals.rolling(window=window).mean()
        rolling_sharpe = returns.rolling(window=window).mean() / returns.rolling(window=window).std()
        
        return {
            'precision': rolling_precision,
            'recall': rolling_recall,
            'f1': rolling_f1,
            'returns': rolling_returns,
            'trade_frequency': rolling_trade_freq,
            'sharpe': rolling_sharpe
        }
    
    def run_backtest(self, model, X_test, y_test, data, threshold=0.5):
        """Run complete backtest"""
        print(f"🔄 Running backtest with threshold {threshold}...")
        
        # Generate signals
        signals, y_proba = self.generate_signals(model, X_test, threshold)
        
        # Calculate returns
        returns = self.calculate_returns(data, signals)
        
        # Calculate metrics
        metrics = self.calculate_trading_metrics(y_test, signals, returns, y_proba)
        
        # Calculate rolling metrics
        rolling_metrics = self.calculate_rolling_metrics(y_test, signals, returns)
        
        # Create results dictionary
        results = {
            'threshold': threshold,
            'signals': signals,
            'returns': returns,
            'y_true': y_test,
            'y_proba': y_proba,
            'metrics': metrics,
            'rolling_metrics': rolling_metrics
        }
        
        print(f"✅ Backtest completed")
        print(f"   Total Return: {metrics['total_return']:.4f}")
        print(f"   Cumulative Return: {metrics['cumulative_return']:.4f}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"   Win Rate: {metrics['win_rate']:.4f}")
        print(f"   Trade Frequency: {metrics['trade_frequency']:.3f}")
        print(f"   F1 Score: {metrics['f1']:.4f}")
        
        return results
    
    def plot_backtest_results(self, results, save_path="tuned_model_backtest_results.png"):
        """Create comprehensive backtest plots"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Tuned Model Backtest Results (Threshold={results["threshold"]})', fontsize=16, fontweight='bold')
        
        # 1. Cumulative Returns
        cumulative_returns = (1 + results['returns']).cumprod()
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns, linewidth=2, color='blue')
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        cumulative = (1 + results['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(drawdown.index, drawdown, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rolling Precision
        if len(results['rolling_metrics']['precision']) > 0:
            x_vals = range(len(results['rolling_metrics']['precision']))
            axes[1, 0].plot(x_vals, results['rolling_metrics']['precision'], linewidth=1, alpha=0.7)
            axes[1, 0].set_title('Rolling Precision')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Rolling Sharpe
        if len(results['rolling_metrics']['sharpe']) > 0:
            axes[1, 1].plot(results['rolling_metrics']['sharpe'].index, results['rolling_metrics']['sharpe'], linewidth=1, alpha=0.7)
            axes[1, 1].set_title('Rolling Sharpe Ratio')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Trade Frequency
        if len(results['rolling_metrics']['trade_frequency']) > 0:
            axes[2, 0].plot(results['rolling_metrics']['trade_frequency'].index, results['rolling_metrics']['trade_frequency'], linewidth=1, alpha=0.7)
            axes[2, 0].set_title('Rolling Trade Frequency')
            axes[2, 0].set_ylabel('Trade Frequency')
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Prediction Confidence Distribution
        axes[2, 1].hist(results['y_proba'], bins=50, alpha=0.7, color='green')
        axes[2, 1].set_title('Prediction Confidence Distribution')
        axes[2, 1].set_xlabel('Predicted Probability')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Backtest plots saved as {save_path}")
    
    def create_backtest_summary(self, results, save_path="tuned_model_backtest_summary.csv"):
        """Create detailed backtest summary"""
        metrics = results['metrics']
        
        summary_data = {
            'Metric': [
                'Total Return', 'Cumulative Return', 'Sharpe Ratio', 'Max Drawdown',
                'Win Rate', 'Volatility', 'VaR (95%)', 'CVaR (95%)',
                'Trade Frequency', 'Avg Trade Return', 'Avg Win', 'Avg Loss',
                'Profit Factor', 'Precision', 'Recall', 'F1 Score', 'Accuracy',
                'Avg Confidence', 'Confidence Std'
            ],
            'Value': [
                metrics['total_return'], metrics['cumulative_return'], metrics['sharpe_ratio'],
                metrics['max_drawdown'], metrics['win_rate'], metrics['volatility'],
                metrics['var_95'], metrics['cvar_95'], metrics['trade_frequency'],
                metrics['avg_trade_return'], metrics['avg_win'], metrics['avg_loss'],
                metrics['profit_factor'], metrics['precision'], metrics['recall'],
                metrics['f1'], metrics['accuracy'], metrics['avg_confidence'],
                metrics['confidence_std']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(save_path, index=False)
        print(f"✅ Backtest summary saved as {save_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("TUNED MODEL BACKTEST SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False, float_format='%.4f'))
        print("="*80)
        
        return summary_df

def main():
    parser = argparse.ArgumentParser(description="Backtest tuned LightGBM model")
    parser.add_argument("--data", type=str, default="merged_with_regime_features.csv", 
                       help="Path to data file")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Prediction threshold for trading signals")
    parser.add_argument("--rolling_window", type=int, default=200, 
                       help="Rolling window size for metrics")
    parser.add_argument("--transaction_cost", type=float, default=0.0, 
                       help="Transaction cost as decimal")
    args = parser.parse_args()
    
    # Initialize backtest engine
    backtest = TunedModelBacktest(
        transaction_cost=args.transaction_cost,
        rolling_window=args.rolling_window
    )
    
    # Load model and data
    model, X_test, y_test, data = backtest.load_tuned_model_and_data(args.data)
    
    if model is None:
        return
    
    # Run backtest
    results = backtest.run_backtest(model, X_test, y_test, data, args.threshold)
    
    # Create plots
    backtest.plot_backtest_results(results)
    
    # Create summary
    backtest.create_backtest_summary(results)
    
    print("\n🎉 Tuned model backtest completed successfully!")

if __name__ == "__main__":
    main() 