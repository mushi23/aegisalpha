#!/usr/bin/env python3
"""
Optimized Paper Trading System
Fast parameter optimization and trade analysis
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

class OptimizedPaperTrading:
    def __init__(self):
        self.results = {}
        self.best_params = {}
        
    def load_data_and_models(self):
        """Load data and models for optimized trading."""
        print("🔄 Loading data and models...")
        
        # Load model
        try:
            model = joblib.load('models_future_return_5_regression/xgboost_future_return_5_regression.pkl')
            print("✅ Loaded XGBoost regression model")
        except Exception as e:
            print(f"❌ Could not load model: {e}")
            return None, None, None
        
        # Load scaler
        try:
            scaler = joblib.load('feature_scaler.pkl')
            print("✅ Loaded feature scaler")
        except Exception as e:
            print(f"❌ Could not load scaler: {e}")
            return None, None, None
        
        # Load data (use last 3 months for faster optimization)
        try:
            df = pd.read_csv("all_currencies_with_indicators_updated.csv")
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Filter to last 3 months for faster processing
            three_months_ago = df['datetime'].max() - timedelta(days=90)
            df = df[df['datetime'] >= three_months_ago].copy()
            
            print(f"✅ Loaded data: {len(df)} rows, {df['pair'].nunique()} pairs (last 3 months)")
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
        
        return model, scaler, trading_data, feature_cols
    
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
    
    def generate_signals(self, df: pd.DataFrame, model, scaler: StandardScaler, 
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
        
        # Generate regression signals
        xgb_pred = model.predict(X_scaled)[0]
        confidence = abs(xgb_pred)
        
        if confidence > confidence_threshold:
            signal = 1 if xgb_pred > 0 else -1
            return {
                'signal': signal,
                'confidence': confidence,
                'predicted_return': xgb_pred,
                'timestamp': datetime.now()
            }
        
        return {}
    
    def run_simulation_with_params(self, data: pd.DataFrame, model, scaler: StandardScaler, 
                                 feature_cols: List[str], params: Dict) -> Dict:
        """Run simulation with specific parameters."""
        
        # Initialize trading state
        initial_capital = params.get('initial_capital', 10000)
        current_capital = initial_capital
        positions = {}
        trade_history = []
        daily_trades = {}
        equity_curve = []
        
        # Configuration parameters
        max_position_size = params.get('max_position_size', 0.1)
        max_daily_trades = params.get('max_daily_trades', 3)
        slippage = params.get('slippage', 0.002)
        transaction_cost = params.get('transaction_cost', 0.001)
        confidence_threshold = params.get('confidence_threshold', 0.002)
        stop_loss = params.get('stop_loss', 0.02)
        take_profit = params.get('take_profit', 0.04)
        step_size = params.get('step_size', 5)
        
        # Process data chronologically
        for i in range(30, len(data), step_size):
            current_data = data.iloc[:i+1]
            current_price = current_data.iloc[-1]['close']
            current_time = current_data.iloc[-1]['datetime']
            current_pair = current_data.iloc[-1]['pair']
            
            # Check exit conditions for existing positions
            for pair, position in list(positions.items()):
                if pair == current_pair:
                    entry_price = position['entry_price']
                    position_size = position['size']
                    
                    # Calculate current P&L
                    if position['direction'] == 'long':
                        pnl = (current_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - current_price) / entry_price
                    
                    # Check stop loss and take profit
                    if pnl <= -stop_loss or pnl >= take_profit:
                        # Close position
                        trade_return = pnl * position_size
                        current_capital += trade_return
                        
                        trade_history.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'pair': pair,
                            'direction': position['direction'],
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'size': position_size,
                            'pnl': trade_return,
                            'return_pct': pnl * 100,
                            'hold_time': (current_time - position['entry_time']).total_seconds() / 3600  # hours
                        })
                        
                        del positions[pair]
            
            # Check daily trade limit
            current_date = current_time.date()
            if current_date not in daily_trades:
                daily_trades[current_date] = 0
            
            if daily_trades[current_date] >= max_daily_trades:
                continue
            
            # Generate signals if no position in current pair
            if current_pair not in positions:
                signals = self.generate_signals(
                    current_data, model, scaler, feature_cols, confidence_threshold
                )
                
                if signals:
                    signal = signals['signal']
                    confidence = signals['confidence']
                    
                    # Calculate position size
                    position_size = min(
                        max_position_size * current_capital,
                        current_capital * 0.1  # Max 10% per trade
                    )
                    
                    # Open position
                    direction = 'long' if signal == 1 else 'short'
                    positions[current_pair] = {
                        'direction': direction,
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'size': position_size,
                        'confidence': confidence
                    }
                    
                    daily_trades[current_date] += 1
            
            # Record equity
            total_equity = current_capital
            for pair, position in positions.items():
                if pair == current_pair:
                    entry_price = position['entry_price']
                    position_size = position['size']
                    
                    if position['direction'] == 'long':
                        pnl = (current_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - current_price) / entry_price
                    
                    total_equity += pnl * position_size
            
            equity_curve.append({
                'datetime': current_time,
                'equity': total_equity,
                'capital': current_capital
            })
        
        # Calculate performance metrics
        return self.calculate_performance_metrics(trade_history, equity_curve, initial_capital, params)
    
    def calculate_performance_metrics(self, trade_history: List, equity_curve: List, 
                                    initial_capital: float, params: Dict) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not trade_history:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'avg_trade_return': 0,
                'profit_factor': 0,
                'avg_hold_time': 0,
                'params': params
            }
        
        # Basic metrics
        total_trades = len(trade_history)
        winning_trades = len([t for t in trade_history if t['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Returns
        total_pnl = sum(t['pnl'] for t in trade_history)
        total_return = total_pnl / initial_capital
        total_return_pct = total_return * 100
        
        # Average trade return
        avg_trade_return = total_pnl / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in trade_history if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trade_history if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average hold time
        avg_hold_time = np.mean([t['hold_time'] for t in trade_history]) if trade_history else 0
        
        # Sharpe ratio (simplified)
        if equity_curve:
            equity_returns = []
            for i in range(1, len(equity_curve)):
                prev_equity = equity_curve[i-1]['equity']
                curr_equity = equity_curve[i]['equity']
                if prev_equity > 0:
                    ret = (curr_equity - prev_equity) / prev_equity
                    equity_returns.append(ret)
            
            if equity_returns:
                avg_return = np.mean(equity_returns)
                std_return = np.std(equity_returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        if equity_curve:
            peak = equity_curve[0]['equity']
            max_drawdown = 0
            
            for point in equity_curve:
                equity = point['equity']
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_trade_return': avg_trade_return,
            'profit_factor': profit_factor,
            'avg_hold_time': avg_hold_time,
            'trade_history': trade_history,
            'equity_curve': equity_curve,
            'params': params
        }
    
    def optimize_parameters(self, data: pd.DataFrame, model, scaler: StandardScaler, 
                          feature_cols: List[str]) -> Dict:
        """Optimize trading parameters using grid search."""
        print("🔍 Optimizing parameters...")
        
        # Parameter grid for optimization
        param_grid = {
            'confidence_threshold': [0.001, 0.002, 0.003, 0.005],
            'stop_loss': [0.015, 0.02, 0.025, 0.03],
            'take_profit': [0.03, 0.04, 0.05, 0.06],
            'max_position_size': [0.05, 0.1, 0.15],
            'max_daily_trades': [2, 3, 5]
        }
        
        best_result = None
        best_score = -float('inf')
        all_results = []
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        print(f"Testing {total_combinations} parameter combinations...")
        
        # Test each combination
        for i, combination in enumerate(np.array(np.meshgrid(*param_values)).T.reshape(-1, len(param_names))):
            params = dict(zip(param_names, combination))
            params.update({
                'initial_capital': 10000,
                'slippage': 0.002,
                'transaction_cost': 0.001,
                'step_size': 5
            })
            
            # Run simulation
            result = self.run_simulation_with_params(data, model, scaler, feature_cols, params)
            
            # Calculate score (Sharpe ratio + win rate bonus)
            score = result['sharpe_ratio'] + (result['win_rate'] * 0.5)
            
            all_results.append({
                'params': params,
                'result': result,
                'score': score
            })
            
            if score > best_score and result['total_trades'] >= 5:  # Minimum trades requirement
                best_score = score
                best_result = result
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{total_combinations} combinations tested")
        
        print(f"✅ Parameter optimization completed!")
        print(f"Best score: {best_score:.3f}")
        
        return best_result, all_results
    
    def analyze_trade_patterns(self, trade_history: List) -> Dict:
        """Analyze trade patterns to understand performance."""
        if not trade_history:
            return {}
        
        df = pd.DataFrame(trade_history)
        
        # Basic analysis
        analysis = {
            'total_trades': len(df),
            'winning_trades': len(df[df['pnl'] > 0]),
            'losing_trades': len(df[df['pnl'] < 0]),
            'win_rate': len(df[df['pnl'] > 0]) / len(df),
            'avg_win': df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0,
            'avg_loss': df[df['pnl'] < 0]['pnl'].mean() if len(df[df['pnl'] < 0]) > 0 else 0,
            'largest_win': df['pnl'].max(),
            'largest_loss': df['pnl'].min(),
            'avg_hold_time': df['hold_time'].mean(),
            'pair_performance': df.groupby('pair')['pnl'].sum().to_dict(),
            'direction_performance': df.groupby('direction')['pnl'].sum().to_dict()
        }
        
        return analysis
    
    def run_optimized_trading(self):
        """Run the complete optimized trading system."""
        print("🚀 Starting Optimized Paper Trading System")
        print("=" * 60)
        
        # Load data and models
        model, scaler, data, feature_cols = self.load_data_and_models()
        if model is None or scaler is None or data is None:
            print("❌ Failed to load required data/models")
            return
        
        # Run parameter optimization
        start_time = datetime.now()
        best_result, all_results = self.optimize_parameters(data, model, scaler, feature_cols)
        end_time = datetime.now()
        
        if best_result is None:
            print("❌ No valid results found")
            return
        
        # Analyze trade patterns
        trade_analysis = self.analyze_trade_patterns(best_result['trade_history'])
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 OPTIMIZED PAPER TRADING RESULTS")
        print("=" * 60)
        print(f"⏱️  Optimization time: {end_time - start_time}")
        print(f"💰 Initial capital: $10,000")
        print(f"📈 Total return: {best_result['total_return_pct']:.2f}%")
        print(f"📊 Sharpe ratio: {best_result['sharpe_ratio']:.3f}")
        print(f"📉 Max drawdown: {best_result['max_drawdown']:.2%}")
        print(f"🎯 Win rate: {best_result['win_rate']:.1%}")
        print(f"🔄 Total trades: {best_result['total_trades']}")
        print(f"📊 Avg trade return: ${best_result['avg_trade_return']:.2f}")
        print(f"💹 Profit factor: {best_result['profit_factor']:.2f}")
        print(f"⏰ Avg hold time: {best_result['avg_hold_time']:.1f} hours")
        
        print(f"\n🔧 Best Parameters:")
        for key, value in best_result['params'].items():
            print(f"  {key}: {value}")
        
        # Trade pattern analysis
        if trade_analysis:
            print(f"\n📊 Trade Pattern Analysis:")
            print(f"  Winning trades: {trade_analysis['winning_trades']}/{trade_analysis['total_trades']}")
            print(f"  Average win: ${trade_analysis['avg_win']:.2f}")
            print(f"  Average loss: ${trade_analysis['avg_loss']:.2f}")
            print(f"  Largest win: ${trade_analysis['largest_win']:.2f}")
            print(f"  Largest loss: ${trade_analysis['largest_loss']:.2f}")
            
            print(f"\n📈 Performance by Currency Pair:")
            for pair, pnl in trade_analysis['pair_performance'].items():
                print(f"  {pair}: ${pnl:.2f}")
            
            print(f"\n📈 Performance by Direction:")
            for direction, pnl in trade_analysis['direction_performance'].items():
                print(f"  {direction}: ${pnl:.2f}")
        
        # Save results
        best_trades_df = pd.DataFrame(best_result['trade_history'])
        if not best_trades_df.empty:
            best_trades_df.to_csv('optimized_trading_results.csv', index=False)
            print(f"\n✅ Best results saved to: optimized_trading_results.csv")
        
        # Save all results for comparison
        results_summary = []
        for result_data in all_results:
            result = result_data['result']
            summary = {
                'confidence_threshold': result['params']['confidence_threshold'],
                'stop_loss': result['params']['stop_loss'],
                'take_profit': result['params']['take_profit'],
                'max_position_size': result['params']['max_position_size'],
                'max_daily_trades': result['params']['max_daily_trades'],
                'total_return_pct': result['total_return_pct'],
                'sharpe_ratio': result['sharpe_ratio'],
                'win_rate': result['win_rate'],
                'total_trades': result['total_trades'],
                'max_drawdown': result['max_drawdown'],
                'score': result_data['score']
            }
            results_summary.append(summary)
        
        results_df = pd.DataFrame(results_summary)
        results_df.to_csv('parameter_optimization_results.csv', index=False)
        print(f"✅ All parameter results saved to: parameter_optimization_results.csv")
        
        # Create visualizations
        self.create_optimization_visualizations(results_df, best_result)
        
        return best_result, all_results
    
    def create_optimization_visualizations(self, results_df: pd.DataFrame, best_result: Dict):
        """Create visualizations for optimization results."""
        print("📊 Creating visualizations...")
        
        # Parameter sensitivity plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Parameter Optimization Results', fontsize=16)
        
        # Confidence threshold
        axes[0, 0].scatter(results_df['confidence_threshold'], results_df['sharpe_ratio'], alpha=0.6)
        axes[0, 0].set_xlabel('Confidence Threshold')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].set_title('Confidence Threshold vs Sharpe Ratio')
        
        # Stop loss
        axes[0, 1].scatter(results_df['stop_loss'], results_df['sharpe_ratio'], alpha=0.6)
        axes[0, 1].set_xlabel('Stop Loss')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_title('Stop Loss vs Sharpe Ratio')
        
        # Take profit
        axes[0, 2].scatter(results_df['take_profit'], results_df['sharpe_ratio'], alpha=0.6)
        axes[0, 2].set_xlabel('Take Profit')
        axes[0, 2].set_ylabel('Sharpe Ratio')
        axes[0, 2].set_title('Take Profit vs Sharpe Ratio')
        
        # Win rate
        axes[1, 0].scatter(results_df['confidence_threshold'], results_df['win_rate'], alpha=0.6)
        axes[1, 0].set_xlabel('Confidence Threshold')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].set_title('Confidence Threshold vs Win Rate')
        
        # Total return
        axes[1, 1].scatter(results_df['max_position_size'], results_df['total_return_pct'], alpha=0.6)
        axes[1, 1].set_xlabel('Max Position Size')
        axes[1, 1].set_ylabel('Total Return (%)')
        axes[1, 1].set_title('Position Size vs Total Return')
        
        # Max drawdown
        axes[1, 2].scatter(results_df['max_daily_trades'], results_df['max_drawdown'], alpha=0.6)
        axes[1, 2].set_xlabel('Max Daily Trades')
        axes[1, 2].set_ylabel('Max Drawdown')
        axes[1, 2].set_title('Daily Trades vs Max Drawdown')
        
        plt.tight_layout()
        plt.savefig('parameter_optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Best result equity curve
        if best_result['equity_curve']:
            equity_df = pd.DataFrame(best_result['equity_curve'])
            equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])
            
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df['datetime'], equity_df['equity'])
            plt.title('Optimized Paper Trading - Best Parameters Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('optimized_trading_equity.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Charts saved to: parameter_optimization_analysis.png, optimized_trading_equity.png")

def main():
    """Main function to run optimized paper trading."""
    trader = OptimizedPaperTrading()
    results = trader.run_optimized_trading()
    
    if results:
        print("\n🎉 Optimized paper trading completed successfully!")
    else:
        print("\n❌ Optimized paper trading failed!")

if __name__ == "__main__":
    main() 