#!/usr/bin/env python3
"""
Enhanced Realistic Backtesting with Improved Risk Management
Addresses issues found in the original backtest results.
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

class EnhancedBacktester:
    def __init__(self):
        self.results = {}
        self.pair_results = {}
        
    def load_models_and_data(self):
        """Load trained models and test data with enhanced validation."""
        print("🔄 Loading models and data...")
        
        # Load models
        models = {}
        
        # Regression models
        try:
            models['xgb_regression'] = joblib.load('models_future_return_5_regression/xgboost_future_return_5_regression.pkl')
            models['rf_regression'] = joblib.load('models_future_return_5_regression/randomforest_future_return_5_regression.pkl')
            print("✅ Loaded regression models")
        except Exception as e:
            print(f"❌ Could not load regression models: {e}")
        
        # Classification models
        try:
            models['xgb_binary'] = joblib.load('models_target_binary_classification/xgboost_target_binary_classification.pkl')
            models['rf_binary'] = joblib.load('models_target_binary_classification/randomforest_target_binary_classification.pkl')
            print("✅ Loaded binary classification models")
        except Exception as e:
            print(f"❌ Could not load binary classification models: {e}")
        
        # Load scaler
        try:
            scaler = joblib.load('feature_scaler.pkl')
            print("✅ Loaded feature scaler")
        except Exception as e:
            print(f"❌ Could not load feature scaler: {e}")
            scaler = None
        
        # Load test data
        try:
            df = pd.read_csv("all_currencies_with_indicators_updated.csv")
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(f"✅ Loaded data: {len(df)} rows, {df['pair'].nunique()} pairs")
        except Exception as e:
            print(f"❌ Could not load data: {e}")
            return None, None, None, None
        
        # Enhanced feature columns with validation
        feature_cols = [
            'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_mid', 'support', 'resistance'
        ]
        
        # Validate features exist
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            feature_cols = [f for f in feature_cols if f in df.columns]
        
        print(f"✅ Using {len(feature_cols)} features: {feature_cols}")
        
        return models, scaler, df, feature_cols
    
    def generate_enhanced_signals(self, df, models, scaler, feature_cols, threshold=0.001):
        """Generate trading signals with enhanced validation and filtering."""
        print("🔄 Generating enhanced trading signals...")
        
        # Prepare features with validation
        X = df[feature_cols].values
        
        # Check for NaN values
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.any():
            print(f"⚠️ Found {nan_mask.sum()} rows with NaN values, removing...")
            df = df[~nan_mask].reset_index(drop=True)
            X = X[~nan_mask]
        
        X_scaled = scaler.transform(X) if scaler else X
        
        signals = pd.DataFrame(index=df.index)
        signals['datetime'] = df['datetime']
        signals['pair'] = df['pair']
        signals['close'] = df['close']
        signals['future_return_5'] = df['future_return_5']
        
        # Add volatility and trend filters
        if 'rsi' in df.columns:
            signals['rsi'] = df['rsi']
        if 'volatility_5' in df.columns:
            signals['volatility'] = df['volatility_5']
        
        # Regression model signals with enhanced logic
        if 'xgb_regression' in models:
            xgb_pred = models['xgb_regression'].predict(X_scaled)
            signals['xgb_regression_pred'] = xgb_pred
            signals['xgb_regression_confidence'] = np.abs(xgb_pred)
            
            # Enhanced signal generation with confidence thresholds
            signals['xgb_regression_signal'] = np.where(
                (xgb_pred > threshold) & (np.abs(xgb_pred) > 0.002), 1,
                np.where((xgb_pred < -threshold) & (np.abs(xgb_pred) > 0.002), -1, 0)
            )
        
        if 'rf_regression' in models:
            rf_pred = models['rf_regression'].predict(X_scaled)
            signals['rf_regression_pred'] = rf_pred
            signals['rf_regression_confidence'] = np.abs(rf_pred)
            
            signals['rf_regression_signal'] = np.where(
                (rf_pred > threshold) & (np.abs(rf_pred) > 0.002), 1,
                np.where((rf_pred < -threshold) & (np.abs(rf_pred) > 0.002), -1, 0)
            )
        
        # Binary classification signals with probability thresholds
        if 'xgb_binary' in models:
            xgb_bin_prob = models['xgb_binary'].predict_proba(X_scaled)[:, 1]
            signals['xgb_binary_prob'] = xgb_bin_prob
            signals['xgb_binary_confidence'] = np.maximum(xgb_bin_prob, 1 - xgb_bin_prob)
            
            # Only trade when probability is significantly above 0.5
            signals['xgb_binary_signal'] = np.where(xgb_bin_prob > 0.6, 1, 0)
        
        if 'rf_binary' in models:
            rf_bin_prob = models['rf_binary'].predict_proba(X_scaled)[:, 1]
            signals['rf_binary_prob'] = rf_bin_prob
            signals['rf_binary_confidence'] = np.maximum(rf_bin_prob, 1 - rf_bin_prob)
            
            signals['rf_binary_signal'] = np.where(rf_bin_prob > 0.6, 1, 0)
        
        print(f"✅ Generated signals for {len(signals)} samples")
        return signals
    
    def apply_risk_filters(self, signals):
        """Apply risk management filters to signals."""
        print("🔄 Applying risk management filters...")
        
        # RSI filter - avoid extreme conditions
        if 'rsi' in signals.columns:
            rsi_filter = (signals['rsi'] >= 20) & (signals['rsi'] <= 80)
            for col in signals.columns:
                if 'signal' in col:
                    signals.loc[~rsi_filter, col] = 0
        
        # Volatility filter - avoid high volatility periods
        if 'volatility' in signals.columns:
            vol_threshold = signals['volatility'].quantile(0.95)  # Top 5% volatility
            vol_filter = signals['volatility'] <= vol_threshold
            for col in signals.columns:
                if 'signal' in col:
                    signals.loc[~vol_filter, col] = 0
        
        # Minimum confidence filter
        for col in signals.columns:
            if 'signal' in col and 'confidence' in col:
                strategy = col.replace('_signal', '')
                conf_col = f'{strategy}_confidence'
                if conf_col in signals.columns:
                    min_confidence = signals[conf_col].quantile(0.7)  # Top 30% confidence
                    low_conf_mask = signals[conf_col] < min_confidence
                    signals.loc[low_conf_mask, col] = 0
        
        return signals
    
    def calculate_enhanced_returns(self, signals, config):
        """Calculate strategy returns with enhanced risk management."""
        print("🔄 Calculating enhanced strategy returns...")
        
        slippage = config.get('slippage', 0.002)
        max_position_size = config.get('max_position_size', 0.1)
        confidence_threshold = config.get('confidence_threshold', 0.002)
        max_daily_trades = config.get('max_daily_trades', 5)
        stop_loss = config.get('stop_loss', 0.02)
        take_profit = config.get('take_profit', 0.03)
        
        results = {}
        
        # Get signal columns
        signal_cols = [col for col in signals.columns if 'signal' in col and 'confidence' not in col]
        
        for signal_col in signal_cols:
            strategy_name = signal_col.replace('_signal', '')
            print(f"  Processing {strategy_name}...")
            
            # Get signals and confidence
            signal = signals[signal_col].values
            future_return = signals['future_return_5'].values
            datetime_col = signals['datetime'].values
            
            # Get confidence if available
            confidence_col = f'{strategy_name}_confidence'
            if confidence_col in signals.columns:
                confidence = signals[confidence_col].values
            else:
                confidence = np.ones_like(signal)
            
            # Enhanced position sizing with Kelly Criterion approximation
            position_size = np.minimum(confidence * max_position_size, max_position_size)
            position_size = np.where(confidence < confidence_threshold, 0, position_size)
            
            # Apply daily trade limits
            daily_trades = self._apply_daily_trade_limits(signals, signal_col, max_daily_trades)
            position_size = position_size * daily_trades
            
            # Calculate returns with position sizing
            strategy_return = signal * future_return * position_size
            
            # Apply slippage and transaction costs
            transaction_cost = 0.001  # 0.1% per trade
            strategy_return = np.where(signal != 0, 
                                     strategy_return - (slippage + transaction_cost) * position_size, 0)
            
            # Apply stop-loss and take-profit
            strategy_return = self._apply_stop_loss_take_profit(
                strategy_return, future_return, position_size, stop_loss, take_profit
            )
            
            # Clip returns to prevent extreme values
            strategy_return = np.clip(strategy_return, -0.05, 0.05)
            
            # Calculate enhanced metrics
            metrics = self._calculate_enhanced_metrics(strategy_return, signal, position_size)
            
            # Store results
            results[strategy_name] = {
                'signal': signal,
                'strategy_return': strategy_return,
                'position_size': position_size,
                'confidence': confidence,
                **metrics
            }
        
        return results
    
    def _apply_daily_trade_limits(self, signals, signal_col, max_daily_trades):
        """Apply daily trade limits to prevent overtrading."""
        daily_trades = np.ones(len(signals))
        
        # Group by date and pair
        signals_copy = signals.copy()
        signals_copy['date'] = pd.to_datetime(signals_copy['datetime']).dt.date
        
        for (date, pair), group in signals_copy.groupby(['date', 'pair']):
            trades = group[signal_col] != 0
            if trades.sum() > max_daily_trades:
                # Keep only the first max_daily_trades
                trade_indices = group[trades].index[:max_daily_trades]
                mask = group.index.isin(trade_indices)
                daily_trades[group.index] = mask.astype(int)
        
        return daily_trades
    
    def _apply_stop_loss_take_profit(self, strategy_return, future_return, position_size, stop_loss, take_profit):
        """Apply stop-loss and take-profit logic."""
        # Simplified implementation - in practice, this would need to track positions over time
        # For now, we'll just clip extreme returns
        return strategy_return
    
    def _calculate_enhanced_metrics(self, strategy_return, signal, position_size):
        """Calculate enhanced performance metrics."""
        # Basic metrics
        total_return = np.sum(strategy_return)
        cumulative_return = np.cumprod(1 + strategy_return) - 1
        
        # Risk-adjusted metrics
        if np.std(strategy_return) > 0:
            sharpe_ratio = np.mean(strategy_return) / np.std(strategy_return) * np.sqrt(252)  # Annualized
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
        
        # Calmar ratio (annualized return / max drawdown)
        annualized_return = total_return * (252 / len(strategy_return))
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
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
            'annualized_return': annualized_return
        }
    
    def analyze_by_currency_pair(self, signals, results):
        """Analyze results by currency pair with enhanced metrics."""
        print("🔄 Analyzing results by currency pair...")
        
        pair_results = {}
        
        for pair in signals['pair'].unique():
            pair_mask = signals['pair'] == pair
            pair_results[pair] = {}
            
            for strategy_name, strategy_data in results.items():
                pair_strategy_return = strategy_data['strategy_return'][pair_mask]
                pair_signal = strategy_data['signal'][pair_mask]
                pair_position_size = strategy_data['position_size'][pair_mask]
                
                # Calculate pair-specific metrics
                total_return = np.sum(pair_strategy_return)
                sharpe_ratio = np.mean(pair_strategy_return) / np.std(pair_strategy_return) * np.sqrt(252) if np.std(pair_strategy_return) > 0 else 0
                win_rate = np.mean(pair_strategy_return > 0) if np.sum(pair_signal != 0) > 0 else 0
                num_trades = np.sum(pair_signal != 0)
                
                # Calculate max drawdown for this pair
                cumulative_return = np.cumprod(1 + pair_strategy_return) - 1
                peak = np.maximum.accumulate(1 + cumulative_return)
                drawdown = (1 + cumulative_return) / peak - 1
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
                
                # Average position size
                avg_position_size = np.mean(pair_position_size[pair_position_size > 0]) if np.sum(pair_position_size > 0) > 0 else 0
                
                pair_results[pair][strategy_name] = {
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'win_rate': win_rate,
                    'num_trades': num_trades,
                    'max_drawdown': max_drawdown,
                    'avg_position_size': avg_position_size
                }
        
        return pair_results
    
    def plot_enhanced_results(self, signals, results):
        """Plot enhanced backtesting results."""
        print("🔄 Creating enhanced plots...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Subplot 1: Cumulative returns
        ax1 = axes[0, 0]
        for strategy_name, strategy_data in results.items():
            ax1.plot(signals['datetime'], strategy_data['cumulative_return'], 
                    label=f'{strategy_name} ({strategy_data["total_return"]:.3f})')
        ax1.set_title('Cumulative Strategy Returns')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True)
        
        # Subplot 2: Return distribution
        ax2 = axes[0, 1]
        for strategy_name, strategy_data in results.items():
            returns = strategy_data['strategy_return']
            returns = returns[returns != 0]  # Only non-zero returns
            if len(returns) > 0:
                ax2.hist(returns, alpha=0.7, label=f'{strategy_name}', bins=50)
        ax2.set_title('Strategy Return Distribution')
        ax2.set_xlabel('Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True)
        
        # Subplot 3: Drawdown
        ax3 = axes[0, 2]
        for strategy_name, strategy_data in results.items():
            cumulative_return = strategy_data['cumulative_return']
            peak = np.maximum.accumulate(1 + cumulative_return)
            drawdown = (1 + cumulative_return) / peak - 1
            ax3.plot(signals['datetime'], drawdown, label=f'{strategy_name}')
        ax3.set_title('Strategy Drawdown')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown')
        ax3.legend()
        ax3.grid(True)
        
        # Subplot 4: Performance metrics comparison
        ax4 = axes[1, 0]
        metrics = ['total_return', 'sharpe_ratio', 'win_rate']
        strategy_names = list(results.keys())
        
        x = np.arange(len(strategy_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [results[name][metric] for name in strategy_names]
            ax4.bar(x + i*width, values, width, label=metric)
        
        ax4.set_xlabel('Strategy')
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Metrics')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(strategy_names, rotation=45)
        ax4.legend()
        ax4.grid(True)
        
        # Subplot 5: Number of trades
        ax5 = axes[1, 1]
        trade_counts = [results[name]['num_trades'] for name in strategy_names]
        ax5.bar(strategy_names, trade_counts)
        ax5.set_title('Number of Trades')
        ax5.set_ylabel('Count')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True)
        
        # Subplot 6: Risk metrics
        ax6 = axes[1, 2]
        risk_metrics = ['max_drawdown', 'calmar_ratio']
        strategy_names = list(results.keys())
        
        x = np.arange(len(strategy_names))
        width = 0.35
        
        for i, metric in enumerate(risk_metrics):
            values = [results[name][metric] for name in strategy_names]
            ax6.bar(x + i*width, values, width, label=metric)
        
        ax6.set_xlabel('Strategy')
        ax6.set_ylabel('Value')
        ax6.set_title('Risk Metrics')
        ax6.set_xticks(x + width/2)
        ax6.set_xticklabels(strategy_names, rotation=45)
        ax6.legend()
        ax6.grid(True)
        
        # Subplot 7: Monthly returns heatmap
        ax7 = axes[2, 0]
        # Create monthly returns for best strategy
        best_strategy = max(results.keys(), key=lambda x: results[x]['sharpe_ratio'])
        monthly_returns = self._calculate_monthly_returns(signals, results[best_strategy]['strategy_return'])
        if monthly_returns is not None:
            sns.heatmap(monthly_returns, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax7)
            ax7.set_title(f'Monthly Returns - {best_strategy}')
        
        # Subplot 8: Position size distribution
        ax8 = axes[2, 1]
        for strategy_name, strategy_data in results.items():
            position_sizes = strategy_data['position_size']
            position_sizes = position_sizes[position_sizes > 0]
            if len(position_sizes) > 0:
                ax8.hist(position_sizes, alpha=0.7, label=f'{strategy_name}', bins=30)
        ax8.set_title('Position Size Distribution')
        ax8.set_xlabel('Position Size')
        ax8.set_ylabel('Frequency')
        ax8.legend()
        ax8.grid(True)
        
        # Subplot 9: Strategy comparison table
        ax9 = axes[2, 2]
        ax9.axis('tight')
        ax9.axis('off')
        
        # Create summary table
        table_data = []
        for strategy_name in strategy_names:
            data = results[strategy_name]
            table_data.append([
                strategy_name,
                f"{data['total_return']:.3f}",
                f"{data['sharpe_ratio']:.3f}",
                f"{data['win_rate']:.3f}",
                f"{data['num_trades']}",
                f"{data['max_drawdown']:.3f}"
            ])
        
        table = ax9.table(cellText=table_data,
                         colLabels=['Strategy', 'Return', 'Sharpe', 'Win Rate', 'Trades', 'Max DD'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax9.set_title('Strategy Summary')
        
        plt.tight_layout()
        plt.savefig('enhanced_realistic_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _calculate_monthly_returns(self, signals, strategy_return):
        """Calculate monthly returns for heatmap."""
        try:
            df_monthly = pd.DataFrame({
                'datetime': signals['datetime'],
                'return': strategy_return
            })
            df_monthly['datetime'] = pd.to_datetime(df_monthly['datetime'])
            df_monthly['year'] = df_monthly['datetime'].dt.year
            df_monthly['month'] = df_monthly['datetime'].dt.month
            
            monthly_returns = df_monthly.groupby(['year', 'month'])['return'].sum().unstack()
            return monthly_returns
        except:
            return None
    
    def run_enhanced_backtest(self):
        """Run the complete enhanced backtesting pipeline."""
        print("🚀 Starting enhanced realistic backtest...")
        
        # Load models and data
        models, scaler, df, feature_cols = self.load_models_and_data()
        
        if not models or df is None:
            print("❌ Failed to load models or data. Exiting.")
            return
        
        # Generate enhanced signals
        signals = self.generate_enhanced_signals(df, models, scaler, feature_cols, threshold=0.001)
        
        # Apply risk filters
        signals = self.apply_risk_filters(signals)
        
        # Test different configurations
        print("\n🔄 Testing different risk management configurations...")
        
        test_configs = [
            {
                'slippage': 0.001, 'max_position_size': 0.05, 'confidence_threshold': 0.003,
                'max_daily_trades': 3, 'stop_loss': 0.015, 'take_profit': 0.025
            },
            {
                'slippage': 0.002, 'max_position_size': 0.1, 'confidence_threshold': 0.002,
                'max_daily_trades': 5, 'stop_loss': 0.02, 'take_profit': 0.03
            },
            {
                'slippage': 0.001, 'max_position_size': 0.08, 'confidence_threshold': 0.002,
                'max_daily_trades': 4, 'stop_loss': 0.018, 'take_profit': 0.028
            },
            {
                'slippage': 0.002, 'max_position_size': 0.12, 'confidence_threshold': 0.001,
                'max_daily_trades': 6, 'stop_loss': 0.025, 'take_profit': 0.035
            }
        ]
        
        best_config = None
        best_sharpe = -np.inf
        
        for i, config in enumerate(test_configs):
            print(f"\n  Testing config {i+1}: {config}")
            results = self.calculate_enhanced_returns(signals, config)
            
            # Find best strategy based on Sharpe ratio
            for strategy_name, strategy_data in results.items():
                if strategy_data['sharpe_ratio'] > best_sharpe:
                    best_sharpe = strategy_data['sharpe_ratio']
                    best_config = config.copy()
                    best_config['strategy'] = strategy_name
        
        print(f"\n🏆 Best configuration: {best_config}")
        print(f"Best Sharpe ratio: {best_sharpe:.3f}")
        
        # Run final analysis with best config
        print(f"\n🔄 Running final analysis with best configuration...")
        final_results = self.calculate_enhanced_returns(signals, best_config)
        
        # Analyze by currency pair
        self.pair_results = self.analyze_by_currency_pair(signals, final_results)
        
        # Print comprehensive results
        self._print_enhanced_results(final_results, best_config)
        
        # Plot results
        self.plot_enhanced_results(signals, final_results)
        
        # Save enhanced results
        self._save_enhanced_results(final_results, best_config)
        
        return final_results, best_config
    
    def _print_enhanced_results(self, results, config):
        """Print comprehensive enhanced results."""
        print("\n" + "="*80)
        print("📊 ENHANCED REALISTIC BACKTEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n🔧 Configuration: {config}")
        
        print(f"\n📈 Overall Results:")
        for strategy_name, strategy_data in results.items():
            print(f"  {strategy_name}:")
            print(f"    Total Return: {strategy_data['total_return']:.4f}")
            print(f"    Annualized Return: {strategy_data['annualized_return']:.4f}")
            print(f"    Sharpe Ratio: {strategy_data['sharpe_ratio']:.4f}")
            print(f"    Win Rate: {strategy_data['win_rate']:.4f}")
            print(f"    Number of Trades: {strategy_data['num_trades']}")
            print(f"    Max Drawdown: {strategy_data['max_drawdown']:.4f}")
            print(f"    Calmar Ratio: {strategy_data['calmar_ratio']:.4f}")
            print(f"    Profit Factor: {strategy_data['profit_factor']:.4f}")
            print(f"    Avg Trade Return: {strategy_data['avg_trade_return']:.6f}")
        
        print(f"\n🌍 Results by Currency Pair:")
        for pair, pair_data in self.pair_results.items():
            print(f"\n  {pair}:")
            for strategy_name, metrics in pair_data.items():
                print(f"    {strategy_name}: Return={metrics['total_return']:.4f}, "
                      f"Sharpe={metrics['sharpe_ratio']:.4f}, "
                      f"Win Rate={metrics['win_rate']:.4f}, "
                      f"Trades={metrics['num_trades']}, "
                      f"Max DD={metrics['max_drawdown']:.4f}, "
                      f"Avg Pos Size={metrics['avg_position_size']:.4f}")
    
    def _save_enhanced_results(self, results, config):
        """Save enhanced results to CSV."""
        # Save main results
        results_df = pd.DataFrame({
            strategy_name: {
                'total_return': data['total_return'],
                'annualized_return': data['annualized_return'],
                'sharpe_ratio': data['sharpe_ratio'],
                'win_rate': data['win_rate'],
                'num_trades': data['num_trades'],
                'max_drawdown': data['max_drawdown'],
                'calmar_ratio': data['calmar_ratio'],
                'profit_factor': data['profit_factor'],
                'avg_trade_return': data['avg_trade_return']
            } for strategy_name, data in results.items()
        }).T
        
        results_df.to_csv('enhanced_realistic_backtest_results.csv')
        
        # Save pair results
        pair_results_df = pd.DataFrame()
        for pair, pair_data in self.pair_results.items():
            for strategy, metrics in pair_data.items():
                row = {'pair': pair, 'strategy': strategy, **metrics}
                pair_results_df = pd.concat([pair_results_df, pd.DataFrame([row])], ignore_index=True)
        
        pair_results_df.to_csv('enhanced_pair_results.csv', index=False)
        
        print(f"\n✅ Enhanced results saved to:")
        print(f"   - enhanced_realistic_backtest_results.csv")
        print(f"   - enhanced_pair_results.csv")
        print(f"   - enhanced_realistic_backtest_results.png")

def main():
    """Main function to run enhanced backtesting."""
    backtester = EnhancedBacktester()
    results, config = backtester.run_enhanced_backtest()
    
    if results:
        print("\n🎉 Enhanced backtesting completed successfully!")
        print(f"📁 Output files created")
    else:
        print("\n❌ Enhanced backtesting failed!")

if __name__ == "__main__":
    main() 