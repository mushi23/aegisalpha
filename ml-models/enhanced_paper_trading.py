#!/usr/bin/env python3
"""
Enhanced Paper Trading System
Features: Longer simulations, parameter optimization, time period testing, portfolio optimization
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import json
import os
import warnings
from typing import Dict, List, Optional, Tuple
from itertools import product
warnings.filterwarnings('ignore')

class EnhancedPaperTrading:
    def __init__(self):
        self.results = {}
        self.parameter_results = {}
        self.portfolio_results = {}
        
    def load_data_and_models(self):
        """Load data and models for enhanced trading."""
        print("🔄 Loading data and models for enhanced trading...")
        
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
            return None, None, None
        
        # Load data
        try:
            df = pd.read_csv("all_currencies_with_indicators_updated.csv")
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(f"✅ Loaded data: {len(df)} rows, {df['pair'].nunique()} pairs")
        except Exception as e:
            print(f"❌ Could not load data: {e}")
            return None, None, None
        
        # Filter to trading hours only
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['hour'] = df['datetime'].dt.hour
        trading_data = df[
            (df['day_of_week'] < 5) &  # Monday to Friday
            (df['hour'] >= 0) & (df['hour'] <= 23)  # 24-hour forex market
        ].copy()
        
        print(f"✅ Filtered to trading hours: {len(trading_data)} rows")
        
        # Feature columns
        feature_cols = [
            'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_mid', 'support', 'resistance'
        ]
        
        return models, scaler, trading_data, feature_cols
    
    def calculate_technical_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for real-time data."""
        df = price_data.copy()
        
        # Basic technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        
        # Support and Resistance
        df['support'] = df['close'].rolling(window=20).min()
        df['resistance'] = df['close'].rolling(window=20).max()
        
        # Volatility
        df['volatility_5'] = df['close'].pct_change().rolling(window=5).std()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, models: Dict, scaler: StandardScaler, 
                        feature_cols: List[str], confidence_threshold: float = 0.002) -> Dict:
        """Generate trading signals with configurable parameters."""
        if df.empty or len(df) < 30:
            return {}
        
        # Calculate features
        df = self.calculate_technical_features(df)
        
        # Get latest data point
        latest_data = df.iloc[-1:]
        
        # Check for required features
        missing_features = [f for f in feature_cols if f not in latest_data.columns]
        if missing_features:
            return {}
        
        # Prepare features for prediction
        X = latest_data[feature_cols].values
        
        # Check for NaN values
        if np.isnan(X).any():
            return {}
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        signals = {}
        
        # Generate regression signals
        if 'xgb_regression' in models:
            xgb_pred = models['xgb_regression'].predict(X_scaled)[0]
            confidence = abs(xgb_pred)
            
            if confidence > confidence_threshold:
                signal = 1 if xgb_pred > 0 else -1
                signals['xgb_regression'] = {
                    'signal': signal,
                    'confidence': confidence,
                    'predicted_return': xgb_pred,
                    'timestamp': datetime.now()
                }
        
        # Generate binary signals
        if 'xgb_binary' in models:
            xgb_bin_prob = models['xgb_binary'].predict_proba(X_scaled)[0, 1]
            confidence = max(xgb_bin_prob, 1 - xgb_bin_prob)
            
            if xgb_bin_prob > 0.6 and confidence > confidence_threshold:
                signals['xgb_binary'] = {
                    'signal': 1,
                    'confidence': confidence,
                    'probability': xgb_bin_prob,
                    'timestamp': datetime.now()
                }
        
        return signals
    
    def run_single_simulation(self, data: pd.DataFrame, models: Dict, scaler: StandardScaler, 
                            feature_cols: List[str], config: Dict) -> Dict:
        """Run a single trading simulation with given configuration."""
        
        # Initialize trading state
        initial_capital = config['initial_capital']
        current_capital = initial_capital
        positions = {}
        trade_history = []
        daily_trades = {}
        equity_curve = []
        
        # Configuration parameters
        max_position_size = config['max_position_size']
        max_daily_trades = config['max_daily_trades']
        slippage = config['slippage']
        transaction_cost = config['transaction_cost']
        confidence_threshold = config['confidence_threshold']
        stop_loss = config['stop_loss']
        take_profit = config['take_profit']
        step_size = config.get('step_size', 5)
        
        # Process data chronologically
        for i in range(30, len(data), step_size):
            current_data = data.iloc[:i+1]
            current_price = current_data.iloc[-1]['close']
            current_time = current_data.iloc[-1]['datetime']
            current_pair = current_data.iloc[-1]['pair']
            
            # Check exit conditions for existing positions
            for pair in list(positions.keys()):
                position = positions[pair]
                
                # Check stop loss and take profit
                if position['side'] == 'buy':
                    if current_price <= position['stop_loss']:
                        # Close position
                        pnl = (current_price - position['entry_price']) * position['quantity']
                        costs = (slippage + transaction_cost) * position['quantity'] * current_price
                        net_pnl = pnl - costs
                        current_capital += net_pnl
                        
                        trade_history.append({
                            'pair': pair,
                            'side': 'sell',
                            'quantity': position['quantity'],
                            'price': current_price,
                            'pnl': net_pnl,
                            'timestamp': current_time,
                            'exit_reason': 'stop_loss'
                        })
                        
                        del positions[pair]
                        
                    elif current_price >= position['take_profit']:
                        # Close position
                        pnl = (current_price - position['entry_price']) * position['quantity']
                        costs = (slippage + transaction_cost) * position['quantity'] * current_price
                        net_pnl = pnl - costs
                        current_capital += net_pnl
                        
                        trade_history.append({
                            'pair': pair,
                            'side': 'sell',
                            'quantity': position['quantity'],
                            'price': current_price,
                            'pnl': net_pnl,
                            'timestamp': current_time,
                            'exit_reason': 'take_profit'
                        })
                        
                        del positions[pair]
            
            # Generate new signals
            signals = self.generate_signals(current_data, models, scaler, feature_cols, confidence_threshold)
            
            # Execute trades if we have signals and meet risk limits
            if signals and current_pair not in positions:
                # Check daily trade limit
                today = current_time.date()
                if today not in daily_trades:
                    daily_trades[today] = {}
                if current_pair not in daily_trades[today]:
                    daily_trades[today][current_pair] = 0
                
                if daily_trades[today][current_pair] < max_daily_trades:
                    # Execute trade
                    for signal_type, signal in signals.items():
                        # Calculate position size
                        base_position_size = current_capital * max_position_size
                        confidence = signal.get('confidence', 0.5)
                        position_size = base_position_size * confidence
                        
                        # Calculate trade quantity
                        trade_quantity = position_size / current_price
                        
                        # Calculate costs
                        total_cost = (slippage + transaction_cost) * position_size
                        
                        # Execute trade
                        trade_side = 'buy' if signal['signal'] > 0 else 'sell'
                        
                        positions[current_pair] = {
                            'side': trade_side,
                            'quantity': trade_quantity,
                            'entry_price': current_price,
                            'entry_time': current_time,
                            'stop_loss': current_price * (1 - stop_loss) if signal['signal'] > 0 else current_price * (1 + stop_loss),
                            'take_profit': current_price * (1 + take_profit) if signal['signal'] > 0 else current_price * (1 - take_profit)
                        }
                        
                        # Update capital
                        current_capital -= total_cost
                        
                        # Update daily trade count
                        daily_trades[today][current_pair] += 1
                        
                        # Add to trade history
                        trade_history.append({
                            'pair': current_pair,
                            'side': trade_side,
                            'quantity': trade_quantity,
                            'price': current_price,
                            'costs': total_cost,
                            'timestamp': current_time,
                            'signal_type': signal_type
                        })
                        
                        break  # Only execute one trade per iteration
            
            # Update equity curve
            total_pnl = sum([t.get('pnl', 0) for t in trade_history])
            equity = initial_capital + total_pnl
            equity_curve.append({
                'timestamp': current_time,
                'equity': equity,
                'capital': current_capital,
                'open_positions': len(positions)
            })
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(trade_history, equity_curve, initial_capital)
        
        return {
            'config': config,
            'trade_history': trade_history,
            'equity_curve': equity_curve,
            'metrics': metrics
        }
    
    def calculate_performance_metrics(self, trade_history: List, equity_curve: List, initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not trade_history:
            return {}
        
        # Filter completed trades (with P&L)
        completed_trades = [t for t in trade_history if 'pnl' in t]
        
        if not completed_trades:
            return {}
        
        # Basic metrics
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t['pnl'] > 0])
        losing_trades = len([t for t in completed_trades if t['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum([t['pnl'] for t in completed_trades])
        avg_win = np.mean([t['pnl'] for t in completed_trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in completed_trades if t['pnl'] < 0]) if losing_trades > 0 else 0
        
        # Risk metrics
        returns = [t['pnl'] / initial_capital for t in completed_trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Drawdown calculation
        equity_values = [e['equity'] for e in equity_curve]
        if equity_values:
            peak = np.maximum.accumulate(equity_values)
            drawdown = (equity_values - peak) / peak
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0
        
        # Profit factor
        gross_profit = sum([t['pnl'] for t in completed_trades if t['pnl'] > 0])
        gross_loss = abs(sum([t['pnl'] for t in completed_trades if t['pnl'] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Total return
        total_return = (initial_capital + total_pnl - initial_capital) / initial_capital
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'final_capital': initial_capital + total_pnl
        }
    
    def run_parameter_optimization(self, data: pd.DataFrame, models: Dict, scaler: StandardScaler, 
                                 feature_cols: List[str]) -> Dict:
        """Run parameter optimization across different configurations."""
        print("🔄 Running parameter optimization...")
        
        # Define parameter ranges
        param_ranges = {
            'confidence_threshold': [0.001, 0.002, 0.003, 0.005],
            'max_position_size': [0.05, 0.1, 0.15, 0.2],
            'max_daily_trades': [3, 5, 7, 10],
            'stop_loss': [0.015, 0.02, 0.025, 0.03],
            'take_profit': [0.025, 0.03, 0.035, 0.04]
        }
        
        # Base configuration
        base_config = {
            'initial_capital': 50000,
            'slippage': 0.002,
            'transaction_cost': 0.001,
            'step_size': 5
        }
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        results = []
        
        # Test each parameter combination
        for i, param_combo in enumerate(product(*param_values)):
            config = base_config.copy()
            for j, param_name in enumerate(param_names):
                config[param_name] = param_combo[j]
            
            print(f"  Testing config {i+1}/{len(list(product(*param_values)))}: {config}")
            
            # Run simulation
            result = self.run_single_simulation(data, models, scaler, feature_cols, config)
            
            if result['metrics']:
                results.append({
                    'config': config,
                    'metrics': result['metrics']
                })
        
        # Find best configuration
        if results:
            best_result = max(results, key=lambda x: x['metrics']['sharpe_ratio'])
            print(f"\n🏆 Best configuration: {best_result['config']}")
            print(f"   Sharpe Ratio: {best_result['metrics']['sharpe_ratio']:.3f}")
            print(f"   Total Return: {best_result['metrics']['total_return']:.3f}")
            print(f"   Win Rate: {best_result['metrics']['win_rate']:.3f}")
        
        return results
    
    def run_time_period_analysis(self, data: pd.DataFrame, models: Dict, scaler: StandardScaler, 
                               feature_cols: List[str], config: Dict) -> Dict:
        """Run analysis across different time periods."""
        print("🔄 Running time period analysis...")
        
        # Define time periods
        periods = {
            '2018-2019': ('2018-01-01', '2019-12-31'),
            '2020-2021': ('2020-01-01', '2021-12-31'),
            '2022-2023': ('2022-01-01', '2023-12-31'),
            '2024': ('2024-01-01', '2024-12-31'),
            'Bull Market': ('2020-04-01', '2021-12-31'),
            'Bear Market': ('2022-01-01', '2022-10-31'),
            'High Volatility': ('2020-03-01', '2020-06-30'),
            'Low Volatility': ('2021-07-01', '2021-12-31')
        }
        
        period_results = {}
        
        for period_name, (start_date, end_date) in periods.items():
            print(f"  Testing {period_name}...")
            
            # Filter data for period
            period_mask = (data['datetime'] >= start_date) & (data['datetime'] <= end_date)
            period_data = data[period_mask].copy()
            
            if len(period_data) > 100:  # Minimum data requirement
                result = self.run_single_simulation(period_data, models, scaler, feature_cols, config)
                period_results[period_name] = result['metrics']
            else:
                print(f"    ⚠️ Insufficient data for {period_name}")
        
        return period_results
    
    def run_portfolio_optimization(self, data: pd.DataFrame, models: Dict, scaler: StandardScaler, 
                                 feature_cols: List[str], config: Dict) -> Dict:
        """Run portfolio optimization across multiple currency pairs."""
        print("🔄 Running portfolio optimization...")
        
        # Get unique currency pairs
        pairs = data['pair'].unique()
        
        # Run individual pair analysis
        pair_results = {}
        pair_returns = {}
        
        for pair in pairs:
            print(f"  Analyzing {pair}...")
            pair_data = data[data['pair'] == pair].copy()
            
            if len(pair_data) > 100:
                result = self.run_single_simulation(pair_data, models, scaler, feature_cols, config)
                pair_results[pair] = result['metrics']
                
                # Extract returns for portfolio analysis
                if result['equity_curve']:
                    returns = []
                    for i in range(1, len(result['equity_curve'])):
                        ret = (result['equity_curve'][i]['equity'] - result['equity_curve'][i-1]['equity']) / result['equity_curve'][i-1]['equity']
                        returns.append(ret)
                    pair_returns[pair] = returns
        
        # Calculate portfolio metrics
        if pair_returns:
            # Create returns matrix
            max_length = max(len(returns) for returns in pair_returns.values())
            returns_matrix = []
            
            for pair in pairs:
                if pair in pair_returns:
                    returns = pair_returns[pair]
                    # Pad with zeros if necessary
                    while len(returns) < max_length:
                        returns.append(0)
                    returns_matrix.append(returns[:max_length])
            
            if returns_matrix:
                returns_matrix = np.array(returns_matrix)
                
                # Calculate portfolio metrics
                portfolio_metrics = {
                    'total_pairs': len(pairs),
                    'avg_return': np.mean(returns_matrix),
                    'portfolio_volatility': np.std(returns_matrix),
                    'correlation_matrix': np.corrcoef(returns_matrix),
                    'pair_results': pair_results
                }
                
                return portfolio_metrics
        
        return {'pair_results': pair_results}
    
    def run_enhanced_simulation(self):
        """Run the complete enhanced paper trading analysis."""
        print("🚀 Starting enhanced paper trading analysis...")
        
        # Load data and models
        models, scaler, data, feature_cols = self.load_data_and_models()
        
        if not models or data is None:
            print("❌ Failed to load data or models. Exiting.")
            return
        
        # 1. Run parameter optimization
        print("\n" + "="*60)
        print("📊 PARAMETER OPTIMIZATION")
        print("="*60)
        param_results = self.run_parameter_optimization(data, models, scaler, feature_cols)
        
        # 2. Use best parameters for further analysis
        if param_results:
            best_config = max(param_results, key=lambda x: x['metrics']['sharpe_ratio'])['config']
            print(f"\n🎯 Using best configuration: {best_config}")
            
            # 3. Run time period analysis
            print("\n" + "="*60)
            print("📊 TIME PERIOD ANALYSIS")
            print("="*60)
            time_results = self.run_time_period_analysis(data, models, scaler, feature_cols, best_config)
            
            # 4. Run portfolio optimization
            print("\n" + "="*60)
            print("📊 PORTFOLIO OPTIMIZATION")
            print("="*60)
            portfolio_results = self.run_portfolio_optimization(data, models, scaler, feature_cols, best_config)
            
            # 5. Run longer simulation with best config
            print("\n" + "="*60)
            print("📊 LONGER SIMULATION")
            print("="*60)
            print("Running extended simulation with best configuration...")
            final_result = self.run_single_simulation(data, models, scaler, feature_cols, best_config)
            
            # 6. Save and display results
            self.save_enhanced_results(param_results, time_results, portfolio_results, final_result)
            
            # 7. Create visualizations
            self.create_enhanced_visualizations(param_results, time_results, portfolio_results, final_result)
            
            print("\n🎉 Enhanced paper trading analysis completed!")
        else:
            print("❌ No valid parameter optimization results. Exiting.")
    
    def save_enhanced_results(self, param_results, time_results, portfolio_results, final_result):
        """Save all enhanced results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save parameter optimization results
        if param_results:
            param_df = pd.DataFrame([
                {**r['config'], **r['metrics']} for r in param_results
            ])
            param_df.to_csv(f"enhanced_param_optimization_{timestamp}.csv", index=False)
        
        # Save time period results
        if time_results:
            time_df = pd.DataFrame(time_results).T
            time_df.to_csv(f"enhanced_time_periods_{timestamp}.csv")
        
        # Save portfolio results
        if portfolio_results and 'pair_results' in portfolio_results:
            portfolio_df = pd.DataFrame(portfolio_results['pair_results']).T
            portfolio_df.to_csv(f"enhanced_portfolio_{timestamp}.csv")
        
        # Save final simulation results
        if final_result:
            # Save trade history
            if final_result['trade_history']:
                trades_df = pd.DataFrame(final_result['trade_history'])
                trades_df.to_csv(f"enhanced_final_trades_{timestamp}.csv", index=False)
            
            # Save equity curve
            if final_result['equity_curve']:
                equity_df = pd.DataFrame(final_result['equity_curve'])
                equity_df.to_csv(f"enhanced_final_equity_{timestamp}.csv", index=False)
            
            # Save metrics
            with open(f"enhanced_final_metrics_{timestamp}.json", 'w') as f:
                json.dump(final_result['metrics'], f, indent=2, default=str)
        
        print(f"✅ Enhanced results saved with timestamp: {timestamp}")
    
    def create_enhanced_visualizations(self, param_results, time_results, portfolio_results, final_result):
        """Create comprehensive visualizations for enhanced results."""
        print("🔄 Creating enhanced visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Subplot 1: Parameter optimization - Sharpe ratio
        if param_results:
            ax1 = axes[0, 0]
            sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in param_results]
            ax1.hist(sharpe_ratios, bins=20, alpha=0.7)
            ax1.set_title('Parameter Optimization - Sharpe Ratio Distribution')
            ax1.set_xlabel('Sharpe Ratio')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Time period analysis
        if time_results:
            ax2 = axes[0, 1]
            periods = list(time_results.keys())
            returns = [time_results[p]['total_return'] for p in periods if 'total_return' in time_results[p]]
            ax2.bar(range(len(returns)), returns)
            ax2.set_title('Time Period Analysis - Total Returns')
            ax2.set_xlabel('Time Period')
            ax2.set_ylabel('Total Return')
            ax2.set_xticks(range(len(returns)))
            ax2.set_xticklabels(periods[:len(returns)], rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Portfolio analysis
        if portfolio_results and 'pair_results' in portfolio_results:
            ax3 = axes[0, 2]
            pairs = list(portfolio_results['pair_results'].keys())
            pair_returns = [portfolio_results['pair_results'][p]['total_return'] for p in pairs if 'total_return' in portfolio_results['pair_results'][p]]
            ax3.bar(range(len(pair_returns)), pair_returns)
            ax3.set_title('Portfolio Analysis - Returns by Currency Pair')
            ax3.set_xlabel('Currency Pair')
            ax3.set_ylabel('Total Return')
            ax3.set_xticks(range(len(pair_returns)))
            ax3.set_xticklabels(pairs[:len(pair_returns)], rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Final simulation equity curve
        if final_result and final_result['equity_curve']:
            ax4 = axes[1, 0]
            equity_df = pd.DataFrame(final_result['equity_curve'])
            ax4.plot(equity_df['timestamp'], equity_df['equity'])
            ax4.set_title('Final Simulation - Equity Curve')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Equity')
            ax4.grid(True, alpha=0.3)
        
        # Subplot 5: Trade distribution
        if final_result and final_result['trade_history']:
            ax5 = axes[1, 1]
            trades_df = pd.DataFrame(final_result['trade_history'])
            if 'pnl' in trades_df.columns:
                ax5.hist(trades_df['pnl'], bins=20, alpha=0.7)
                ax5.set_title('Trade P&L Distribution')
                ax5.set_xlabel('P&L')
                ax5.set_ylabel('Frequency')
                ax5.grid(True, alpha=0.3)
        
        # Subplot 6: Performance summary
        ax6 = axes[1, 2]
        ax6.axis('tight')
        ax6.axis('off')
        
        if final_result and final_result['metrics']:
            metrics = final_result['metrics']
            table_data = [
                ['Metric', 'Value'],
                ['Total Return', f"{metrics['total_return']:.3f}"],
                ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.3f}"],
                ['Win Rate', f"{metrics['win_rate']:.3f}"],
                ['Total Trades', f"{metrics['total_trades']}"],
                ['Max Drawdown', f"{metrics['max_drawdown']:.3f}"],
                ['Profit Factor', f"{metrics['profit_factor']:.3f}"]
            ]
            
            table = ax6.table(cellText=table_data, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax6.set_title('Final Performance Summary')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'enhanced_paper_trading_results_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Enhanced visualizations saved as enhanced_paper_trading_results_{timestamp}.png")

def main():
    """Main function to run enhanced paper trading."""
    enhanced_trader = EnhancedPaperTrading()
    enhanced_trader.run_enhanced_simulation()

if __name__ == "__main__":
    main() 