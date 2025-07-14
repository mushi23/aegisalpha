#!/usr/bin/env python3
"""
Amplified Strategy Testing with 20x Signal Amplification
Tests the trading strategy with amplified signals to boost performance.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AmplifiedStrategyTester:
    def __init__(self, amplification_factor=20):
        self.amplification_factor = amplification_factor
        self.results = {}
        
    def load_data_and_models(self):
        """Load data and models for amplified testing."""
        print("🔄 Loading data and models for amplified testing...")
        
        # Load models
        models = {}
        try:
            models['xgb_regression'] = joblib.load('models_future_return_5_regression/xgboost_future_return_5_regression.pkl')
            models['xgb_binary'] = joblib.load('models_target_binary_classification/xgboost_target_binary_classification.pkl')
            print("✅ Loaded XGBoost models")
        except Exception as e:
            print(f"❌ Could not load models: {e}")
            return None, None, None
        
        # Load scaler
        try:
            scaler = joblib.load('feature_scaler.pkl')
            print("✅ Loaded feature scaler")
        except Exception as e:
            print(f"❌ Could not load scaler: {e}")
            scaler = None
        
        # Load data
        try:
            df = pd.read_csv("all_currencies_with_indicators_updated.csv")
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(f"✅ Loaded data: {len(df)} rows, {df['pair'].nunique()} pairs")
        except Exception as e:
            print(f"❌ Could not load data: {e}")
            return None, None, None
        
        # Feature columns
        feature_cols = [
            'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_mid', 'support', 'resistance'
        ]
        
        # Validate features
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            feature_cols = [f for f in feature_cols if f in df.columns]
        
        return models, scaler, df, feature_cols
    
    def generate_amplified_signals(self, df, models, scaler, feature_cols, threshold=0.001):
        """Generate amplified trading signals."""
        print(f"🔄 Generating amplified signals (20x amplification)...")
        
        # Prepare features
        X = df[feature_cols].values
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.any():
            df = df[~nan_mask].reset_index(drop=True)
            X = X[~nan_mask]
        
        X_scaled = scaler.transform(X) if scaler else X
        
        signals = pd.DataFrame(index=df.index)
        signals['datetime'] = df['datetime']
        signals['pair'] = df['pair']
        signals['close'] = df['close']
        signals['future_return_5'] = df['future_return_5']
        
        # Add market condition indicators
        if 'rsi' in df.columns:
            signals['rsi'] = df['rsi']
        if 'volatility_5' in df.columns:
            signals['volatility'] = df['volatility_5']
        elif 'volatility' in df.columns:
            signals['volatility'] = df['volatility']
        
        # Generate amplified signals for each model
        if 'xgb_regression' in models:
            xgb_pred = models['xgb_regression'].predict(X_scaled)
            signals['xgb_regression_pred'] = xgb_pred
            signals['xgb_regression_confidence'] = np.abs(xgb_pred)
            
            # Amplify the signals by 20x
            amplified_signal = np.where(
                (xgb_pred > threshold) & (np.abs(xgb_pred) > 0.002), 
                xgb_pred * self.amplification_factor,
                np.where((xgb_pred < -threshold) & (np.abs(xgb_pred) > 0.002), 
                        xgb_pred * self.amplification_factor, 0)
            )
            
            # Clip amplified signals to reasonable bounds
            amplified_signal = np.clip(amplified_signal, -2.0, 2.0)
            signals['xgb_regression_signal'] = np.sign(amplified_signal)  # Convert to -1, 0, 1
            signals['xgb_regression_amplified_strength'] = np.abs(amplified_signal)
        
        if 'xgb_binary' in models:
            xgb_bin_prob = models['xgb_binary'].predict_proba(X_scaled)[:, 1]
            signals['xgb_binary_prob'] = xgb_bin_prob
            signals['xgb_binary_confidence'] = np.maximum(xgb_bin_prob, 1 - xgb_bin_prob)
            
            # Amplify binary signals
            amplified_prob = xgb_bin_prob * self.amplification_factor
            amplified_prob = np.clip(amplified_prob, 0, 1)  # Keep within [0,1]
            signals['xgb_binary_signal'] = np.where(amplified_prob > 0.6, 1, 0)
            signals['xgb_binary_amplified_strength'] = amplified_prob
        
        return signals
    
    def calculate_amplified_returns(self, signals, config):
        """Calculate strategy returns with amplified signals."""
        print("🔄 Calculating amplified strategy returns...")
        
        slippage = config.get('slippage', 0.002)
        max_position_size = config.get('max_position_size', 0.1)
        confidence_threshold = config.get('confidence_threshold', 0.002)
        
        results = {}
        
        signal_cols = [col for col in signals.columns if 'signal' in col and 'confidence' not in col and 'amplified' not in col]
        
        for signal_col in signal_cols:
            strategy_name = signal_col.replace('_signal', '')
            print(f"  Processing {strategy_name}...")
            
            signal = signals[signal_col].values
            future_return = signals['future_return_5'].values
            
            # Get amplified strength if available
            strength_col = f'{strategy_name}_amplified_strength'
            if strength_col in signals.columns:
                amplified_strength = signals[strength_col].values
            else:
                amplified_strength = np.ones_like(signal)
            
            # Get confidence
            confidence_col = f'{strategy_name}_confidence'
            if confidence_col in signals.columns:
                confidence = signals[confidence_col].values
            else:
                confidence = np.ones_like(signal)
            
            # Enhanced position sizing with amplification
            base_position_size = np.minimum(confidence * max_position_size, max_position_size)
            base_position_size = np.where(confidence < confidence_threshold, 0, base_position_size)
            
            # Apply amplification to position size
            amplified_position_size = base_position_size * amplified_strength
            
            # Cap position size to prevent excessive risk
            max_amplified_position = max_position_size * 2  # Allow up to 2x base position
            amplified_position_size = np.clip(amplified_position_size, 0, max_amplified_position)
            
            # Calculate returns with amplified position sizing
            strategy_return = signal * future_return * amplified_position_size
            
            # Apply costs (proportional to position size)
            transaction_cost = 0.001
            strategy_return = np.where(signal != 0, 
                                     strategy_return - (slippage + transaction_cost) * amplified_position_size, 0)
            
            # Clip returns to prevent extreme values
            strategy_return = np.clip(strategy_return, -0.1, 0.1)
            
            # Calculate enhanced metrics
            metrics = self._calculate_enhanced_metrics(strategy_return, signal, amplified_position_size)
            
            # Store results
            results[strategy_name] = {
                'signal': signal,
                'strategy_return': strategy_return,
                'position_size': amplified_position_size,
                'amplified_strength': amplified_strength,
                **metrics
            }
        
        return results
    
    def _calculate_enhanced_metrics(self, strategy_return, signal, position_size):
        """Calculate enhanced performance metrics."""
        # Basic metrics
        total_return = np.sum(strategy_return)
        cumulative_return = np.cumprod(1 + strategy_return) - 1
        
        # Risk-adjusted metrics
        if np.std(strategy_return) > 0:
            sharpe_ratio = np.mean(strategy_return) / np.std(strategy_return) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Win rate and trade analysis
        trades = strategy_return[strategy_return != 0]
        win_rate = np.mean(trades > 0) if len(trades) > 0 else 0
        num_trades = len(trades)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(1 + cumulative_return)
        drawdown = (1 + cumulative_return) / peak - 1
        max_drawdown = np.min(drawdown)
        
        # Additional metrics
        avg_trade_return = np.mean(trades) if len(trades) > 0 else 0
        profit_factor = np.sum(trades[trades > 0]) / abs(np.sum(trades[trades < 0])) if np.sum(trades[trades < 0]) != 0 else np.inf
        
        # Calmar ratio
        annualized_return = total_return * (252 / len(strategy_return))
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Amplification-specific metrics
        avg_position_size = np.mean(position_size[position_size > 0]) if np.sum(position_size > 0) > 0 else 0
        max_position_size_used = np.max(position_size) if len(position_size) > 0 else 0
        
        return {
            'total_return': total_return,
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'max_drawdown': max_drawdown,
            'avg_trade_return': avg_trade_return,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'annualized_return': annualized_return,
            'avg_position_size': avg_position_size,
            'max_position_size_used': max_position_size_used
        }
    
    def test_amplification_levels(self, base_config):
        """Test different amplification levels."""
        print("🔄 Testing different amplification levels...")
        
        amplification_levels = [1, 2]  # Only test 1x and 2x
        amplification_results = {}
        
        for amp_level in amplification_levels:
            print(f"  Testing {amp_level}x amplification...")
            self.amplification_factor = amp_level
            # Always use the original full feature DataFrame
            amplified_signals = self.generate_amplified_signals(
                self.df, self.models, self.scaler, self.feature_cols
            )
            results = self.calculate_amplified_returns(amplified_signals, base_config)
            amplification_results[f'{amp_level}x'] = results
        
        return amplification_results
    
    def run_amplified_test(self):
        """Run the complete amplified strategy test."""
        print(f"🚀 Starting amplified strategy test ({self.amplification_factor}x amplification)...")
        
        # Load data and models
        self.models, self.scaler, self.df, self.feature_cols = self.load_data_and_models()
        
        if not self.models or self.df is None:
            print("❌ Failed to load data or models. Exiting.")
            return
        
        # Generate amplified signals
        signals = self.generate_amplified_signals(self.df, self.models, self.scaler, self.feature_cols)
        
        # Base configuration
        base_config = {
            'slippage': 0.002,
            'max_position_size': 0.1,
            'confidence_threshold': 0.002
        }
        
        # Test different amplification levels
        print("\n🔄 Testing different amplification levels...")
        amplification_results = self.test_amplification_levels(base_config)
        
        # Run final test with 20x amplification
        print(f"\n🔄 Running final test with {self.amplification_factor}x amplification...")
        self.amplification_factor = 20
        final_signals = self.generate_amplified_signals(self.df, self.models, self.scaler, self.feature_cols)
        final_results = self.calculate_amplified_returns(final_signals, base_config)
        
        # Analyze results
        self.analyze_amplified_results(amplification_results, final_results)
        
        # Create visualizations
        self.plot_amplified_results(amplification_results, final_results)
        
        return final_results, amplification_results
    
    def analyze_amplified_results(self, amplification_results, final_results):
        """Analyze and summarize amplified results."""
        print("\n📊 Analyzing amplified results...")
        
        # Create summary dataframe
        summary_data = []
        
        for amp_level, results in amplification_results.items():
            for strategy_name, metrics in results.items():
                summary_data.append({
                    'amplification': amp_level,
                    'strategy': strategy_name,
                    'total_return': metrics['total_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'win_rate': metrics['win_rate'],
                    'num_trades': metrics['num_trades'],
                    'max_drawdown': metrics['max_drawdown'],
                    'avg_position_size': metrics['avg_position_size'],
                    'max_position_size_used': metrics['max_position_size_used']
                })
        
        self.summary_df = pd.DataFrame(summary_data)
        
        # Print key findings
        print("\n" + "="*80)
        print("📊 AMPLIFIED STRATEGY RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n🎯 Amplification Factor: {self.amplification_factor}x")
        
        # Performance by amplification level
        print(f"\n📈 Performance by Amplification Level:")
        amp_performance = self.summary_df.groupby('amplification').agg({
            'total_return': 'mean',
            'sharpe_ratio': 'mean',
            'win_rate': 'mean',
            'max_drawdown': 'mean'
        }).round(4)
        print(amp_performance)
        
        # Best amplification level
        best_amp = self.summary_df.loc[self.summary_df['sharpe_ratio'].idxmax()]
        print(f"\n🏆 Best Amplification Level:")
        print(f"  {best_amp['amplification']} - {best_amp['strategy']}")
        print(f"  Sharpe: {best_amp['sharpe_ratio']:.4f}")
        print(f"  Return: {best_amp['total_return']:.4f}")
        print(f"  Win Rate: {best_amp['win_rate']:.4f}")
        
        # Risk analysis
        print(f"\n⚠️ Risk Analysis:")
        max_dd = self.summary_df['max_drawdown'].min()
        max_dd_row = self.summary_df.loc[self.summary_df['max_drawdown'].idxmin()]
        print(f"  Worst Drawdown: {max_dd:.4f} at {max_dd_row['amplification']}")
        
        # Position size analysis
        print(f"\n📊 Position Size Analysis:")
        pos_stats = self.summary_df.groupby('amplification').agg({
            'avg_position_size': 'mean',
            'max_position_size_used': 'max'
        }).round(4)
        print(pos_stats)
        
        # Save detailed results
        self.summary_df.to_csv('amplified_strategy_results.csv', index=False)
        print(f"\n✅ Detailed results saved to: amplified_strategy_results.csv")
    
    def plot_amplified_results(self, amplification_results, final_results):
        """Create comprehensive amplified results visualizations."""
        print("🔄 Creating amplified results visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Subplot 1: Return vs Amplification
        ax1 = axes[0, 0]
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            ax1.plot(strategy_data['amplification'], strategy_data['total_return'], 
                    marker='o', label=strategy, linewidth=2)
        ax1.set_title('Total Return vs Amplification Level')
        ax1.set_xlabel('Amplification Factor')
        ax1.set_ylabel('Total Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Sharpe vs Amplification
        ax2 = axes[0, 1]
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            ax2.plot(strategy_data['amplification'], strategy_data['sharpe_ratio'], 
                    marker='s', label=strategy, linewidth=2)
        ax2.set_title('Sharpe Ratio vs Amplification Level')
        ax2.set_xlabel('Amplification Factor')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Win Rate vs Amplification
        ax3 = axes[0, 2]
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            ax3.plot(strategy_data['amplification'], strategy_data['win_rate'], 
                    marker='^', label=strategy, linewidth=2)
        ax3.set_title('Win Rate vs Amplification Level')
        ax3.set_xlabel('Amplification Factor')
        ax3.set_ylabel('Win Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Max Drawdown vs Amplification
        ax4 = axes[1, 0]
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            ax4.plot(strategy_data['amplification'], strategy_data['max_drawdown'], 
                    marker='v', label=strategy, linewidth=2, color='red')
        ax4.set_title('Max Drawdown vs Amplification Level')
        ax4.set_xlabel('Amplification Factor')
        ax4.set_ylabel('Max Drawdown')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Subplot 5: Position Size vs Amplification
        ax5 = axes[1, 1]
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            ax5.plot(strategy_data['amplification'], strategy_data['avg_position_size'], 
                    marker='d', label=strategy, linewidth=2)
        ax5.set_title('Average Position Size vs Amplification Level')
        ax5.set_xlabel('Amplification Factor')
        ax5.set_ylabel('Average Position Size')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Subplot 6: Risk vs Return scatter
        ax6 = axes[1, 2]
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            ax6.scatter(strategy_data['max_drawdown'], strategy_data['total_return'], 
                       alpha=0.7, label=strategy, s=100)
        ax6.set_title('Risk vs Return (All Amplification Levels)')
        ax6.set_xlabel('Max Drawdown')
        ax6.set_ylabel('Total Return')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Subplot 7: Number of trades vs Amplification
        ax7 = axes[2, 0]
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            ax7.plot(strategy_data['amplification'], strategy_data['num_trades'], 
                    marker='o', label=strategy, linewidth=2)
        ax7.set_title('Number of Trades vs Amplification Level')
        ax7.set_xlabel('Amplification Factor')
        ax7.set_ylabel('Number of Trades')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Subplot 8: Performance comparison table
        ax8 = axes[2, 1]
        ax8.axis('tight')
        ax8.axis('off')
        
        # Create summary table for 20x amplification
        table_data = []
        for strategy_name, metrics in final_results.items():
            table_data.append([
                strategy_name,
                f"{metrics['total_return']:.4f}",
                f"{metrics['sharpe_ratio']:.4f}",
                f"{metrics['win_rate']:.4f}",
                f"{metrics['num_trades']}",
                f"{metrics['max_drawdown']:.4f}"
            ])
        
        table = ax8.table(cellText=table_data,
                         colLabels=['Strategy', 'Return', 'Sharpe', 'Win Rate', 'Trades', 'Max DD'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax8.set_title(f'{self.amplification_factor}x Amplification Results')
        
        # Subplot 9: Optimal amplification level
        ax9 = axes[2, 2]
        best_amp_data = self.summary_df.groupby('amplification')['sharpe_ratio'].mean()
        ax9.bar(best_amp_data.index, best_amp_data.values, alpha=0.7)
        ax9.set_title('Average Sharpe Ratio by Amplification Level')
        ax9.set_xlabel('Amplification Factor')
        ax9.set_ylabel('Average Sharpe Ratio')
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('amplified_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Amplified results visualizations saved to: amplified_strategy_results.png")

def main():
    """Main function to run amplified strategy testing."""
    # Test with 20x amplification
    amplified_tester = AmplifiedStrategyTester(amplification_factor=20)
    final_results, amplification_results = amplified_tester.run_amplified_test()
    
    if final_results:
        print("\n🎉 Amplified strategy testing completed successfully!")
        print("📁 Output files:")
        print("   - amplified_strategy_results.csv")
        print("   - amplified_strategy_results.png")
    else:
        print("\n❌ Amplified strategy testing failed!")

if __name__ == "__main__":
    main() 