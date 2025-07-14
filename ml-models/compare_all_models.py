#!/usr/bin/env python3
"""
Comprehensive Model Comparison
Compares tuned model vs previous models to show dramatic improvements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelComparison:
    def __init__(self, transaction_cost=0.0, rolling_window=200):
        """Initialize model comparison"""
        self.transaction_cost = transaction_cost
        self.rolling_window = rolling_window
        
    def load_all_models_and_data(self, data_path):
        """Load all models and prepare data"""
        print("🔄 Loading all models and data...")
        
        models = {}
        feature_sets = {}
        
        # Load tuned model
        try:
            models['tuned'] = joblib.load('lgbm_best_model.pkl')
            print("✅ Tuned model loaded")
            with open('lgbm_hyperparam_tuning_results.csv', 'r') as f:
                pass  # just to check existence
            with open('lgbm_best_model.pkl', 'rb') as _:
                pass
            with open('feature_list_full_technical.txt', 'r') as f:
                features_tuned = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            # Try to load from threshold JSON if available
            try:
                with open('lgbm_best_model_threshold.json', 'r') as f:
                    threshold_info = json.load(f)
                    features_tuned = threshold_info.get('feature_names', features_tuned)
            except:
                pass
            feature_sets['tuned'] = features_tuned
        except Exception as e:
            print(f"❌ Error loading tuned model: {e}")
            models['tuned'] = None
            feature_sets['tuned'] = []
        
        # Load previous LightGBM model
        try:
            models['previous'] = joblib.load('corrective_ai_lgbm.pkl')
            print("✅ Previous LightGBM model loaded")
            with open('corrective_ai_lgbm_threshold.json', 'r') as f:
                threshold_info = json.load(f)
                features_previous = threshold_info.get('feature_names', [])
            feature_sets['previous'] = features_previous
        except Exception as e:
            print(f"❌ Error loading previous model: {e}")
            models['previous'] = None
            feature_sets['previous'] = []
        
        # Load no-bull model
        try:
            models['no_bull'] = joblib.load('corrective_ai_lgbm_nobull.pkl')
            print("✅ No-bull model loaded")
            with open('corrective_ai_lgbm_nobull_threshold.json', 'r') as f:
                threshold_info = json.load(f)
                features_no_bull = threshold_info.get('feature_names', [])
            feature_sets['no_bull'] = features_no_bull
        except Exception as e:
            print(f"❌ Error loading no-bull model: {e}")
            models['no_bull'] = None
            feature_sets['no_bull'] = []
        
        # Load data
        data = pd.read_csv(data_path)
        
        # Create label if needed
        if 'label' not in data.columns and 'return' in data.columns:
            data['label'] = (data['return'] > 0).astype(int)
        
        # Prepare X for each model
        X = {}
        for key in feature_sets:
            features = [f for f in feature_sets[key] if f in data.columns]
            X[key] = data[features]
        y = data['label']
        
        # Train-test split (same as training)
        X_train, X_test, y_train, y_test = train_test_split(
            X['tuned'], y, test_size=0.2, shuffle=False, stratify=None
        )
        test_indices = X_test.index
        X_test_dict = {k: X[k].iloc[test_indices] for k in X}
        
        print(f"✅ Data prepared for all models, {len(X_test)} test samples")
        
        return models, X_test_dict, y_test, data.iloc[test_indices]
    
    def generate_signals(self, model, X_test, threshold=0.5):
        """Generate trading signals"""
        y_proba = model.predict_proba(X_test)[:, 1]
        return (y_proba > threshold).astype(int), y_proba
    
    def calculate_returns(self, data, signals):
        """Calculate strategy returns"""
        if 'return' in data.columns:
            price_returns = data['return']
        elif 'close' in data.columns:
            price_returns = data['close'].pct_change()
        else:
            raise ValueError("Need 'return' or 'close' column in data")
        
        strategy_returns = signals * price_returns
        
        if self.transaction_cost > 0:
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
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Trade metrics
        trade_frequency = signals.mean()
        avg_trade_return = returns[signals == 1].mean() if (signals == 1).sum() > 0 else 0
        avg_win = returns[(returns > 0) & (signals == 1)].mean() if ((returns > 0) & (signals == 1)).sum() > 0 else 0
        avg_loss = returns[(returns < 0) & (signals == 1)].mean() if ((returns < 0) & (signals == 1)).sum() > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
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
            'profit_factor': profit_factor
        }
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def run_model_backtest(self, model, X_test, y_test, data, model_name, threshold=0.5):
        """Run backtest for one model"""
        print(f"🔄 Running backtest for {model_name}...")
        
        if model is None:
            print(f"❌ {model_name} model not available")
            return None
        
        # Generate signals
        signals, y_proba = self.generate_signals(model, X_test, threshold)
        
        # Calculate returns
        returns = self.calculate_returns(data, signals)
        
        # Calculate metrics
        metrics = self.calculate_trading_metrics(y_test, signals, returns, y_proba)
        
        results = {
            'model_name': model_name,
            'threshold': threshold,
            'signals': signals,
            'returns': returns,
            'y_true': y_test,
            'y_proba': y_proba,
            'metrics': metrics
        }
        
        print(f"✅ {model_name} backtest completed")
        print(f"   Total Return: {metrics['total_return']:.4f}")
        print(f"   Cumulative Return: {metrics['cumulative_return']:.4f}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"   F1 Score: {metrics['f1']:.4f}")
        print(f"   Trade Frequency: {metrics['trade_frequency']:.3f}")
        
        return results
    
    def plot_comparison(self, results_list, save_path="all_models_comparison.png"):
        """Create comprehensive comparison plots"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('All Models Comparison', fontsize=16, fontweight='bold')
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, results in enumerate(results_list):
            if results is None:
                continue
                
            color = colors[i % len(colors)]
            label = results['model_name']
            
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
            
            # 3. Rolling Returns (200-period)
            rolling_returns = results['returns'].rolling(window=200).sum()
            axes[1, 0].plot(rolling_returns.index, rolling_returns, 
                           label=label, color=color, linewidth=1, alpha=0.7)
            
            # 4. Rolling Sharpe (200-period)
            rolling_sharpe = results['returns'].rolling(window=200).mean() / results['returns'].rolling(window=200).std()
            axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe, 
                           label=label, color=color, linewidth=1, alpha=0.7)
            
            # 5. Trade Frequency (200-period)
            # Convert signals to pandas Series if needed
            signals = results['signals']
            if isinstance(signals, np.ndarray):
                signals = pd.Series(signals, index=results['returns'].index)
            rolling_trade_freq = signals.rolling(window=200).mean()
            axes[2, 0].plot(rolling_trade_freq.index, rolling_trade_freq, 
                           label=label, color=color, linewidth=1, alpha=0.7)
            
            # 6. Prediction Confidence Distribution
            axes[2, 1].hist(results['y_proba'], bins=50, alpha=0.5, 
                           label=label, color=color)
        
        # Set labels and titles
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Rolling Returns (200-period)')
        axes[1, 0].set_ylabel('Returns')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Rolling Sharpe Ratio (200-period)')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[2, 0].set_title('Rolling Trade Frequency (200-period)')
        axes[2, 0].set_ylabel('Trade Frequency')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].set_title('Prediction Confidence Distribution')
        axes[2, 1].set_xlabel('Predicted Probability')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Comparison plots saved as {save_path}")
    
    def create_comparison_summary(self, results_list, save_path="all_models_comparison_summary.csv"):
        """Create comprehensive comparison summary"""
        comparison_data = []
        
        for results in results_list:
            if results is None:
                continue
                
            metrics = results['metrics']
            comparison_data.append({
                'Model': results['model_name'],
                'Threshold': results['threshold'],
                'Total_Return': metrics['total_return'],
                'Cumulative_Return': metrics['cumulative_return'],
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Max_Drawdown': metrics['max_drawdown'],
                'Win_Rate': metrics['win_rate'],
                'Volatility': metrics['volatility'],
                'VaR_95': metrics['var_95'],
                'Trade_Frequency': metrics['trade_frequency'],
                'Avg_Trade_Return': metrics['avg_trade_return'],
                'Profit_Factor': metrics['profit_factor'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1'],
                'Accuracy': metrics['accuracy']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(save_path, index=False)
        print(f"✅ Comparison summary saved as {save_path}")
        
        # Print summary
        print("\n" + "="*100)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*100)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        print("="*100)
        
        # Calculate improvements
        if len(comparison_data) >= 2:
            tuned = comparison_data[0]  # Assuming tuned is first
            previous = comparison_data[1]  # Assuming previous is second
            
            print(f"\n🚀 TUNED MODEL IMPROVEMENTS:")
            print(f"  Total Return: {tuned['Total_Return']:.4f} vs {previous['Total_Return']:.4f} (+{tuned['Total_Return'] - previous['Total_Return']:.4f})")
            print(f"  Cumulative Return: {tuned['Cumulative_Return']:.4f} vs {previous['Cumulative_Return']:.4f} (+{tuned['Cumulative_Return'] - previous['Cumulative_Return']:.4f})")
            print(f"  Sharpe Ratio: {tuned['Sharpe_Ratio']:.4f} vs {previous['Sharpe_Ratio']:.4f} (+{tuned['Sharpe_Ratio'] - previous['Sharpe_Ratio']:.4f})")
            print(f"  Max Drawdown: {tuned['Max_Drawdown']:.4f} vs {previous['Max_Drawdown']:.4f} ({tuned['Max_Drawdown'] - previous['Max_Drawdown']:.4f})")
            print(f"  F1 Score: {tuned['F1_Score']:.4f} vs {previous['F1_Score']:.4f} (+{tuned['F1_Score'] - previous['F1_Score']:.4f})")
            print(f"  Profit Factor: {tuned['Profit_Factor']:.4f} vs {previous['Profit_Factor']:.4f} (+{tuned['Profit_Factor'] - previous['Profit_Factor']:.4f})")
        
        return comparison_df

def main():
    parser = argparse.ArgumentParser(description="Compare all models")
    parser.add_argument("--data", type=str, default="merged_with_regime_features.csv", 
                       help="Path to data file")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Prediction threshold for trading signals")
    parser.add_argument("--rolling_window", type=int, default=200, 
                       help="Rolling window size for metrics")
    parser.add_argument("--transaction_cost", type=float, default=0.0, 
                       help="Transaction cost as decimal")
    args = parser.parse_args()
    
    # Initialize comparison
    comparison = ModelComparison(
        transaction_cost=args.transaction_cost,
        rolling_window=args.rolling_window
    )
    
    # Load models and data
    models, X_test_dict, y_test, data = comparison.load_all_models_and_data(args.data)
    
    # Run backtests for all models
    results_list = []
    
    # Tuned model
    if models['tuned']:
        results_tuned = comparison.run_model_backtest(
            models['tuned'], X_test_dict['tuned'], y_test, data, "Tuned LightGBM", args.threshold
        )
        results_list.append(results_tuned)
    
    # Previous model
    if models['previous']:
        results_previous = comparison.run_model_backtest(
            models['previous'], X_test_dict['previous'], y_test, data, "Previous LightGBM", args.threshold
        )
        results_list.append(results_previous)
    
    # No-bull model
    if models['no_bull']:
        results_no_bull = comparison.run_model_backtest(
            models['no_bull'], X_test_dict['no_bull'], y_test, data, "No-Bull LightGBM", args.threshold
        )
        results_list.append(results_no_bull)
    
    if not results_list:
        print("❌ No models available for comparison")
        return
    
    # Create comparison plots
    comparison.plot_comparison(results_list)
    
    # Create comparison summary
    comparison.create_comparison_summary(results_list)
    
    print("\n🎉 All models comparison completed successfully!")

if __name__ == "__main__":
    main() 