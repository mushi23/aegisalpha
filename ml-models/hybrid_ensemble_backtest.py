#!/usr/bin/env python3
"""
Hybrid Ensemble Backtest
Combines models with and without bull_prob_hmm using agreement and confidence-weighted strategies.
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

class HybridEnsembleBacktest:
    def __init__(self, transaction_cost=0.0, rolling_window=200):
        """
        Initialize hybrid ensemble backtest
        
        Args:
            transaction_cost: Fixed transaction cost as decimal (e.g., 0.0005 = 5 bps)
            rolling_window: Window size for rolling metrics
        """
        self.transaction_cost = transaction_cost
        self.rolling_window = rolling_window
        
    def load_models_and_data(self, data_path, base_threshold=0.05):
        """Load both models and generate predictions"""
        print("🔄 Loading models and generating predictions...")
        
        # Load models
        model_with_bull = joblib.load('corrective_ai_lgbm.pkl')
        model_without_bull = joblib.load('corrective_ai_lgbm_nobull.pkl')
        
        # Load threshold info
        with open('corrective_ai_lgbm_threshold.json', 'r') as f:
            threshold_info_with = json.load(f)
        with open('corrective_ai_lgbm_nobull_threshold.json', 'r') as f:
            threshold_info_without = json.load(f)
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Create label column if it doesn't exist
        if 'label' not in data.columns and 'return' in data.columns:
            data['label'] = (data['return'] > 0).astype(int)
        
        # Prepare features for both models
        features_with = threshold_info_with['feature_names']
        features_without = threshold_info_without['feature_names']
        
        # Check which features are available
        available_features_with = [f for f in features_with if f in data.columns]
        available_features_without = [f for f in features_without if f in data.columns]
        
        print(f"   Model with bull_prob_hmm: {len(available_features_with)} features")
        print(f"   Model without bull_prob_hmm: {len(available_features_without)} features")
        
        # Generate predictions
        X_with = data[available_features_with]
        X_without = data[available_features_without]
        
        y_proba_with = model_with_bull.predict_proba(X_with)[:, 1]
        y_proba_without = model_without_bull.predict_proba(X_without)[:, 1]
        
        # Generate base signals
        signals_with = (y_proba_with > base_threshold).astype(int)
        signals_without = (y_proba_without > base_threshold).astype(int)
        
        return {
            'data': data,
            'y_true': data['label'],
            'y_proba_with': y_proba_with,
            'y_proba_without': y_proba_without,
            'signals_with': signals_with,
            'signals_without': signals_without,
            'features_with': available_features_with,
            'features_without': available_features_without
        }
    
    def generate_ensemble_signals(self, data_dict, confidence_cutoff=0.8):
        """Generate ensemble signals using both strategies"""
        
        # Strategy 1: Agreement Model
        # Only trade when both models agree on bullish signal
        agreement_signals = ((data_dict['signals_with'] == 1) & 
                           (data_dict['signals_without'] == 1)).astype(int)
        
        # Strategy 2: Confidence-Weighted Model
        # Trade when no_bull_prob predicts 1 AND bull_prob_hmm has high confidence
        confidence_signals = ((data_dict['signals_without'] == 1) & 
                            (data_dict['y_proba_with'] > confidence_cutoff)).astype(int)
        
        return {
            'agreement': agreement_signals,
            'confidence_weighted': confidence_signals
        }
    
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
        
        return {
            'precision': rolling_precision,
            'recall': rolling_recall,
            'f1': rolling_f1,
            'returns': rolling_returns,
            'trade_frequency': rolling_trade_freq
        }
    
    def run_ensemble_backtest(self, data_dict, ensemble_signals, strategy_name):
        """Run backtest for one ensemble strategy"""
        print(f"🔄 Running backtest for {strategy_name}")
        
        # Calculate returns
        returns = self.calculate_returns(data_dict['data'], ensemble_signals)
        
        # Calculate metrics
        metrics = self.calculate_metrics(data_dict['y_true'], ensemble_signals, returns)
        
        # Calculate rolling metrics
        rolling_metrics = self.calculate_rolling_metrics(
            data_dict['y_true'], ensemble_signals, returns
        )
        
        # Create results dictionary
        results = {
            'strategy_name': strategy_name,
            'signals': ensemble_signals,
            'returns': returns,
            'y_true': data_dict['y_true'],
            'metrics': metrics,
            'rolling_metrics': rolling_metrics
        }
        
        print(f"✅ Completed backtest for {strategy_name}")
        print(f"   Total Return: {metrics['total_return']:.4f}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"   Trade Frequency: {metrics['trade_frequency']:.3f}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        
        return results
    
    def plot_ensemble_comparison(self, results_list, save_path="hybrid_ensemble_comparison.png"):
        """Create comprehensive comparison plots"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Hybrid Ensemble Strategies Comparison', fontsize=16, fontweight='bold')
        
        # Colors for different strategies
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, results in enumerate(results_list):
            color = colors[i % len(colors)]
            label = results['strategy_name']
            
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
        print(f"✅ Ensemble comparison plots saved as {save_path}")
    
    def create_ensemble_summary(self, results_list, save_path="hybrid_ensemble_summary.csv"):
        """Create summary table of ensemble results"""
        summary_data = []
        
        for results in results_list:
            metrics = results['metrics']
            summary_data.append({
                'Strategy': results['strategy_name'],
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
        print(f"✅ Ensemble summary saved as {save_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("HYBRID ENSEMBLE SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False, float_format='%.4f'))
        print("="*80)
        
        return summary_df
    
    def analyze_model_agreement(self, data_dict):
        """Analyze agreement between base models"""
        agreement_rate = (data_dict['signals_with'] == data_dict['signals_without']).mean()
        both_bullish = ((data_dict['signals_with'] == 1) & (data_dict['signals_without'] == 1)).mean()
        both_bearish = ((data_dict['signals_with'] == 0) & (data_dict['signals_without'] == 0)).mean()
        
        print(f"\n📊 Model Agreement Analysis:")
        print(f"   Overall Agreement Rate: {agreement_rate:.3f}")
        print(f"   Both Bullish Rate: {both_bullish:.3f}")
        print(f"   Both Bearish Rate: {both_bearish:.3f}")
        print(f"   Disagreement Rate: {1 - agreement_rate:.3f}")
        
        return {
            'agreement_rate': agreement_rate,
            'both_bullish': both_bullish,
            'both_bearish': both_bearish,
            'disagreement_rate': 1 - agreement_rate
        }

def main():
    parser = argparse.ArgumentParser(description="Run hybrid ensemble backtest")
    parser.add_argument("--data", type=str, default="merged_with_regime_features.csv", 
                       help="Path to data file")
    parser.add_argument("--base_threshold", type=float, default=0.05, 
                       help="Base threshold for both models")
    parser.add_argument("--confidence_cutoff", type=float, default=0.8, 
                       help="Confidence cutoff for weighted strategy")
    parser.add_argument("--rolling_window", type=int, default=200, 
                       help="Rolling window size for metrics")
    parser.add_argument("--transaction_cost", type=float, default=0.0, 
                       help="Transaction cost as decimal")
    args = parser.parse_args()
    
    # Initialize ensemble backtest
    ensemble_backtest = HybridEnsembleBacktest(
        transaction_cost=args.transaction_cost,
        rolling_window=args.rolling_window
    )
    
    # Load models and data
    data_dict = ensemble_backtest.load_models_and_data(
        args.data, base_threshold=args.base_threshold
    )
    
    # Analyze model agreement
    agreement_analysis = ensemble_backtest.analyze_model_agreement(data_dict)
    
    # Generate ensemble signals
    ensemble_signals = ensemble_backtest.generate_ensemble_signals(
        data_dict, confidence_cutoff=args.confidence_cutoff
    )
    
    # Run backtests for both strategies
    results_list = []
    
    # Strategy 1: Agreement Model
    results_agreement = ensemble_backtest.run_ensemble_backtest(
        data_dict, ensemble_signals['agreement'], "Agreement Model"
    )
    results_list.append(results_agreement)
    
    # Strategy 2: Confidence-Weighted Model
    results_confidence = ensemble_backtest.run_ensemble_backtest(
        data_dict, ensemble_signals['confidence_weighted'], "Confidence-Weighted Model"
    )
    results_list.append(results_confidence)
    
    # Create comparison plots
    ensemble_backtest.plot_ensemble_comparison(results_list)
    
    # Create summary table
    ensemble_backtest.create_ensemble_summary(results_list)
    
    print("\n🎉 Hybrid ensemble backtest completed successfully!")

if __name__ == "__main__":
    main() 