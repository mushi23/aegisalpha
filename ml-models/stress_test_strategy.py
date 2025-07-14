#!/usr/bin/env python3
"""
Comprehensive Stress Testing Framework for Trading Strategy
Tests strategy performance under various market conditions and scenarios.
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

class StrategyStressTester:
    def __init__(self):
        self.results = {}
        self.stress_scenarios = {}
        
    def load_data_and_models(self):
        """Load data and models for stress testing."""
        print("🔄 Loading data and models for stress testing...")
        
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
    
    def generate_signals(self, df, models, scaler, feature_cols, threshold=0.001):
        """Generate trading signals for stress testing."""
        print("🔄 Generating signals for stress testing...")
        
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
        
        # Generate signals for each model
        if 'xgb_regression' in models:
            xgb_pred = models['xgb_regression'].predict(X_scaled)
            signals['xgb_regression_pred'] = xgb_pred
            signals['xgb_regression_confidence'] = np.abs(xgb_pred)
            signals['xgb_regression_signal'] = np.where(
                (xgb_pred > threshold) & (np.abs(xgb_pred) > 0.002), 1,
                np.where((xgb_pred < -threshold) & (np.abs(xgb_pred) > 0.002), -1, 0)
            )
        
        if 'xgb_binary' in models:
            xgb_bin_prob = models['xgb_binary'].predict_proba(X_scaled)[:, 1]
            signals['xgb_binary_prob'] = xgb_bin_prob
            signals['xgb_binary_confidence'] = np.maximum(xgb_bin_prob, 1 - xgb_bin_prob)
            signals['xgb_binary_signal'] = np.where(xgb_bin_prob > 0.6, 1, 0)
        
        return signals
    
    def calculate_returns(self, signals, config):
        """Calculate strategy returns with given configuration."""
        slippage = config.get('slippage', 0.002)
        max_position_size = config.get('max_position_size', 0.1)
        confidence_threshold = config.get('confidence_threshold', 0.002)
        max_daily_trades = config.get('max_daily_trades', 5)
        
        results = {}
        
        signal_cols = [col for col in signals.columns if 'signal' in col and 'confidence' not in col]
        
        for signal_col in signal_cols:
            strategy_name = signal_col.replace('_signal', '')
            
            signal = signals[signal_col].values
            future_return = signals['future_return_5'].values
            
            confidence_col = f'{strategy_name}_confidence'
            if confidence_col in signals.columns:
                confidence = signals[confidence_col].values
            else:
                confidence = np.ones_like(signal)
            
            # Position sizing
            position_size = np.minimum(confidence * max_position_size, max_position_size)
            position_size = np.where(confidence < confidence_threshold, 0, position_size)
            
            # Apply daily trade limits (simplified)
            # daily_trades = self._apply_daily_trade_limits(signals, signal_col, max_daily_trades)
            # position_size = position_size * daily_trades
            
            # Calculate returns
            strategy_return = signal * future_return * position_size
            
            # Apply costs
            transaction_cost = 0.001
            strategy_return = np.where(signal != 0, 
                                     strategy_return - (slippage + transaction_cost) * position_size, 0)
            
            # Calculate metrics
            total_return = np.sum(strategy_return)
            sharpe_ratio = np.mean(strategy_return) / np.std(strategy_return) * np.sqrt(252) if np.std(strategy_return) > 0 else 0
            win_rate = np.mean(strategy_return > 0) if np.sum(signal != 0) > 0 else 0
            num_trades = np.sum(signal != 0)
            
            cumulative_return = np.cumprod(1 + strategy_return) - 1
            peak = np.maximum.accumulate(1 + cumulative_return)
            drawdown = (1 + cumulative_return) / peak - 1
            max_drawdown = np.min(drawdown)
            
            results[strategy_name] = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'num_trades': num_trades,
                'max_drawdown': max_drawdown,
                'strategy_return': strategy_return
            }
        
        return results
    
    def _apply_daily_trade_limits(self, signals, signal_col, max_daily_trades):
        """Apply daily trade limits."""
        daily_trades = np.ones(len(signals))
        signals_copy = signals.copy()
        signals_copy['date'] = pd.to_datetime(signals_copy['datetime']).dt.date
        
        # Create a mapping from original index to position
        index_to_pos = {idx: pos for pos, idx in enumerate(signals.index)}
        
        for (date, pair), group in signals_copy.groupby(['date', 'pair']):
            trades = group[signal_col] != 0
            if trades.sum() > max_daily_trades:
                # Keep only the first max_daily_trades
                trade_indices = group[trades].index[:max_daily_trades]
                # Set all trades in this group to 0, then set the kept ones to 1
                for idx in group.index:
                    if idx in index_to_pos:
                        daily_trades[index_to_pos[idx]] = 0
                for idx in trade_indices:
                    if idx in index_to_pos:
                        daily_trades[index_to_pos[idx]] = 1
        
        return daily_trades
    
    def stress_test_time_periods(self, signals, base_config):
        """Stress test different time periods."""
        print("🔄 Stress testing different time periods...")
        
        periods = {
            '2018-2019': ('2018-01-01', '2019-12-31'),
            '2020-2021': ('2020-01-01', '2021-12-31'),
            '2022-2023': ('2022-01-01', '2023-12-31'),
            '2024': ('2024-01-01', '2024-12-31'),
            'Bull Market': ('2020-04-01', '2021-12-31'),  # COVID recovery
            'Bear Market': ('2022-01-01', '2022-10-31'),  # 2022 bear market
            'High Volatility': ('2020-03-01', '2020-06-30'),  # COVID crash
            'Low Volatility': ('2021-07-01', '2021-12-31')  # Calm period
        }
        
        period_results = {}
        
        for period_name, (start_date, end_date) in periods.items():
            print(f"  Testing {period_name}...")
            
            # Filter data for period
            period_mask = (signals['datetime'] >= start_date) & (signals['datetime'] <= end_date)
            period_signals = signals[period_mask].copy()
            
            if len(period_signals) > 100:  # Minimum data requirement
                results = self.calculate_returns(period_signals, base_config)
                period_results[period_name] = results
            else:
                print(f"    ⚠️ Insufficient data for {period_name}")
        
        return period_results
    
    def stress_test_market_conditions(self, signals, base_config):
        """Stress test different market conditions."""
        print("🔄 Stress testing different market conditions...")
        
        condition_results = {}
        
        # RSI conditions
        if 'rsi' in signals.columns:
            print("  Testing RSI conditions...")
            
            # Oversold conditions
            oversold_mask = signals['rsi'] < 30
            oversold_signals = signals[oversold_mask].copy()
            if len(oversold_signals) > 100:
                condition_results['RSI_Oversold'] = self.calculate_returns(oversold_signals, base_config)
            
            # Overbought conditions
            overbought_mask = signals['rsi'] > 70
            overbought_signals = signals[overbought_mask].copy()
            if len(overbought_signals) > 100:
                condition_results['RSI_Overbought'] = self.calculate_returns(overbought_signals, base_config)
            
            # Neutral conditions
            neutral_mask = (signals['rsi'] >= 30) & (signals['rsi'] <= 70)
            neutral_signals = signals[neutral_mask].copy()
            if len(neutral_signals) > 100:
                condition_results['RSI_Neutral'] = self.calculate_returns(neutral_signals, base_config)
        
        # Volatility conditions
        if 'volatility' in signals.columns:
            print("  Testing volatility conditions...")
            
            vol_median = signals['volatility'].median()
            
            # High volatility
            high_vol_mask = signals['volatility'] > vol_median * 1.5
            high_vol_signals = signals[high_vol_mask].copy()
            if len(high_vol_signals) > 100:
                condition_results['High_Volatility'] = self.calculate_returns(high_vol_signals, base_config)
            
            # Low volatility
            low_vol_mask = signals['volatility'] < vol_median * 0.5
            low_vol_signals = signals[low_vol_mask].copy()
            if len(low_vol_signals) > 100:
                condition_results['Low_Volatility'] = self.calculate_returns(low_vol_signals, base_config)
        
        return condition_results
    
    def stress_test_parameters(self, signals, base_config):
        """Stress test different parameter combinations."""
        print("🔄 Stress testing different parameters...")
        
        param_results = {}
        
        # Test different slippage levels
        slippage_levels = [0.0005, 0.001, 0.002, 0.005, 0.01]
        for slippage in slippage_levels:
            config = base_config.copy()
            config['slippage'] = slippage
            param_results[f'Slippage_{slippage}'] = self.calculate_returns(signals, config)
        
        # Test different position sizes
        position_sizes = [0.05, 0.1, 0.15, 0.2, 0.25]
        for pos_size in position_sizes:
            config = base_config.copy()
            config['max_position_size'] = pos_size
            param_results[f'PositionSize_{pos_size}'] = self.calculate_returns(signals, config)
        
        # Test different confidence thresholds
        confidence_levels = [0.001, 0.002, 0.003, 0.005, 0.01]
        for conf in confidence_levels:
            config = base_config.copy()
            config['confidence_threshold'] = conf
            param_results[f'Confidence_{conf}'] = self.calculate_returns(signals, config)
        
        return param_results
    
    def stress_test_currency_pairs(self, signals, base_config):
        """Stress test individual currency pairs."""
        print("🔄 Stress testing individual currency pairs...")
        
        pair_results = {}
        
        for pair in signals['pair'].unique():
            print(f"  Testing {pair}...")
            pair_mask = signals['pair'] == pair
            pair_signals = signals[pair_mask].copy()
            
            if len(pair_signals) > 100:
                pair_results[pair] = self.calculate_returns(pair_signals, base_config)
        
        return pair_results
    
    def stress_test_regime_changes(self, signals, base_config):
        """Stress test regime changes and market shifts."""
        print("🔄 Stress testing regime changes...")
        
        regime_results = {}
        
        # Split data into quarters
        signals_copy = signals.copy()
        signals_copy['quarter'] = pd.to_datetime(signals_copy['datetime']).dt.to_period('Q')
        
        for quarter in signals_copy['quarter'].unique():
            quarter_mask = signals_copy['quarter'] == quarter
            quarter_signals = signals_copy[quarter_mask].copy()
            
            if len(quarter_signals) > 100:
                quarter_name = str(quarter)
                regime_results[f'Q{quarter_name}'] = self.calculate_returns(quarter_signals, base_config)
        
        return regime_results
    
    def run_comprehensive_stress_test(self):
        """Run comprehensive stress testing."""
        print("🚀 Starting comprehensive stress testing...")
        
        # Load data and models
        models, scaler, df, feature_cols = self.load_data_and_models()
        
        if not models or df is None:
            print("❌ Failed to load data or models. Exiting.")
            return
        
        # Generate signals
        signals = self.generate_signals(df, models, scaler, feature_cols)
        
        # Base configuration (from enhanced backtest)
        base_config = {
            'slippage': 0.002,
            'max_position_size': 0.12,
            'confidence_threshold': 0.001,
            'max_daily_trades': 6
        }
        
        # Run all stress tests
        print("\n🔄 Running comprehensive stress tests...")
        
        # 1. Time period stress test
        self.stress_scenarios['time_periods'] = self.stress_test_time_periods(signals, base_config)
        
        # 2. Market conditions stress test
        self.stress_scenarios['market_conditions'] = self.stress_test_market_conditions(signals, base_config)
        
        # 3. Parameter stress test
        self.stress_scenarios['parameters'] = self.stress_test_parameters(signals, base_config)
        
        # 4. Currency pair stress test
        self.stress_scenarios['currency_pairs'] = self.stress_test_currency_pairs(signals, base_config)
        
        # 5. Regime changes stress test
        self.stress_scenarios['regime_changes'] = self.stress_test_regime_changes(signals, base_config)
        
        # Analyze and report results
        self.analyze_stress_test_results()
        
        # Create stress test visualizations
        self.plot_stress_test_results()
        
        return self.stress_scenarios
    
    def analyze_stress_test_results(self):
        """Analyze and summarize stress test results."""
        print("\n📊 Analyzing stress test results...")
        
        # Create summary dataframe
        summary_data = []
        
        for scenario_type, scenarios in self.stress_scenarios.items():
            for scenario_name, results in scenarios.items():
                for strategy_name, metrics in results.items():
                    summary_data.append({
                        'scenario_type': scenario_type,
                        'scenario_name': scenario_name,
                        'strategy': strategy_name,
                        'total_return': metrics['total_return'],
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'win_rate': metrics['win_rate'],
                        'num_trades': metrics['num_trades'],
                        'max_drawdown': metrics['max_drawdown']
                    })
        
        self.summary_df = pd.DataFrame(summary_data)
        
        # Print key findings
        print("\n" + "="*80)
        print("📊 STRESS TEST SUMMARY")
        print("="*80)
        
        # Overall performance
        print(f"\n📈 Overall Performance Summary:")
        overall_stats = self.summary_df.groupby('strategy').agg({
            'total_return': ['mean', 'std', 'min', 'max'],
            'sharpe_ratio': ['mean', 'std', 'min', 'max'],
            'win_rate': ['mean', 'std', 'min', 'max']
        }).round(4)
        print(overall_stats)
        
        # Worst case scenarios
        print(f"\n⚠️ Worst Case Scenarios:")
        worst_return = self.summary_df.loc[self.summary_df['total_return'].idxmin()]
        worst_sharpe = self.summary_df.loc[self.summary_df['sharpe_ratio'].idxmin()]
        worst_drawdown = self.summary_df.loc[self.summary_df['max_drawdown'].idxmin()]
        
        print(f"  Worst Return: {worst_return['scenario_name']} - {worst_return['strategy']} ({worst_return['total_return']:.4f})")
        print(f"  Worst Sharpe: {worst_sharpe['scenario_name']} - {worst_sharpe['strategy']} ({worst_sharpe['sharpe_ratio']:.4f})")
        print(f"  Worst Drawdown: {worst_drawdown['scenario_name']} - {worst_drawdown['strategy']} ({worst_drawdown['max_drawdown']:.4f})")
        
        # Best case scenarios
        print(f"\n🏆 Best Case Scenarios:")
        best_return = self.summary_df.loc[self.summary_df['total_return'].idxmax()]
        best_sharpe = self.summary_df.loc[self.summary_df['sharpe_ratio'].idxmax()]
        
        print(f"  Best Return: {best_return['scenario_name']} - {best_return['strategy']} ({best_return['total_return']:.4f})")
        print(f"  Best Sharpe: {best_sharpe['scenario_name']} - {best_sharpe['strategy']} ({best_sharpe['sharpe_ratio']:.4f})")
        
        # Scenario type analysis
        print(f"\n📊 Performance by Scenario Type:")
        scenario_stats = self.summary_df.groupby('scenario_type').agg({
            'total_return': 'mean',
            'sharpe_ratio': 'mean',
            'win_rate': 'mean'
        }).round(4)
        print(scenario_stats)
        
        # Save detailed results
        self.summary_df.to_csv('stress_test_results.csv', index=False)
        print(f"\n✅ Detailed results saved to: stress_test_results.csv")
    
    def plot_stress_test_results(self):
        """Create comprehensive stress test visualizations."""
        print("🔄 Creating stress test visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Subplot 1: Return distribution across all scenarios
        ax1 = axes[0, 0]
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]['total_return']
            ax1.hist(strategy_data, alpha=0.7, label=strategy, bins=20)
        ax1.set_title('Return Distribution Across Scenarios')
        ax1.set_xlabel('Total Return')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Sharpe ratio distribution
        ax2 = axes[0, 1]
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]['sharpe_ratio']
            ax2.hist(strategy_data, alpha=0.7, label=strategy, bins=20)
        ax2.set_title('Sharpe Ratio Distribution')
        ax2.set_xlabel('Sharpe Ratio')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Win rate distribution
        ax3 = axes[0, 2]
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]['win_rate']
            ax3.hist(strategy_data, alpha=0.7, label=strategy, bins=20)
        ax3.set_title('Win Rate Distribution')
        ax3.set_xlabel('Win Rate')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Performance by scenario type
        ax4 = axes[1, 0]
        scenario_means = self.summary_df.groupby('scenario_type')['total_return'].mean()
        ax4.bar(scenario_means.index, scenario_means.values)
        ax4.set_title('Average Return by Scenario Type')
        ax4.set_xlabel('Scenario Type')
        ax4.set_ylabel('Average Return')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Subplot 5: Risk vs Return scatter
        ax5 = axes[1, 1]
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            ax5.scatter(strategy_data['max_drawdown'], strategy_data['total_return'], 
                       alpha=0.7, label=strategy, s=50)
        ax5.set_title('Risk vs Return')
        ax5.set_xlabel('Max Drawdown')
        ax5.set_ylabel('Total Return')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Subplot 6: Sharpe vs Win Rate scatter
        ax6 = axes[1, 2]
        for strategy in self.summary_df['strategy'].unique():
            strategy_data = self.summary_df[self.summary_df['strategy'] == strategy]
            ax6.scatter(strategy_data['win_rate'], strategy_data['sharpe_ratio'], 
                       alpha=0.7, label=strategy, s=50)
        ax6.set_title('Win Rate vs Sharpe Ratio')
        ax6.set_xlabel('Win Rate')
        ax6.set_ylabel('Sharpe Ratio')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Subplot 7: Time period performance
        ax7 = axes[2, 0]
        time_periods = self.summary_df[self.summary_df['scenario_type'] == 'time_periods']
        if not time_periods.empty:
            time_period_means = time_periods.groupby('scenario_name')['total_return'].mean()
            ax7.bar(range(len(time_period_means)), time_period_means.values)
            ax7.set_title('Performance by Time Period')
            ax7.set_xlabel('Time Period')
            ax7.set_ylabel('Average Return')
            ax7.set_xticks(range(len(time_period_means)))
            ax7.set_xticklabels(time_period_means.index, rotation=45)
            ax7.grid(True, alpha=0.3)
        
        # Subplot 8: Currency pair performance
        ax8 = axes[2, 1]
        currency_pairs = self.summary_df[self.summary_df['scenario_type'] == 'currency_pairs']
        if not currency_pairs.empty:
            pair_means = currency_pairs.groupby('scenario_name')['total_return'].mean()
            ax8.bar(range(len(pair_means)), pair_means.values)
            ax8.set_title('Performance by Currency Pair')
            ax8.set_xlabel('Currency Pair')
            ax8.set_ylabel('Average Return')
            ax8.set_xticks(range(len(pair_means)))
            ax8.set_xticklabels(pair_means.index, rotation=45)
            ax8.grid(True, alpha=0.3)
        
        # Subplot 9: Parameter sensitivity
        ax9 = axes[2, 2]
        parameters = self.summary_df[self.summary_df['scenario_type'] == 'parameters']
        if not parameters.empty:
            param_means = parameters.groupby('scenario_name')['total_return'].mean()
            ax9.bar(range(len(param_means)), param_means.values)
            ax9.set_title('Parameter Sensitivity')
            ax9.set_xlabel('Parameter Setting')
            ax9.set_ylabel('Average Return')
            ax9.set_xticks(range(len(param_means)))
            ax9.set_xticklabels(param_means.index, rotation=45)
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('stress_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Stress test visualizations saved to: stress_test_results.png")

def main():
    """Main function to run stress testing."""
    stress_tester = StrategyStressTester()
    results = stress_tester.run_comprehensive_stress_test()
    
    if results:
        print("\n🎉 Stress testing completed successfully!")
        print("📁 Output files:")
        print("   - stress_test_results.csv")
        print("   - stress_test_results.png")
    else:
        print("\n❌ Stress testing failed!")

if __name__ == "__main__":
    main() 