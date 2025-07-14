#!/usr/bin/env python3
"""
Test Optimized Parameters
Test the best parameters on longer timeframe for validation
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from typing import List, Dict

class OptimizedParameterTester:
    def __init__(self):
        self.results = {}
        
    def load_data_and_models(self):
        """Load data and models for testing."""
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
        
        # Load data (use last 12 months for validation)
        try:
            df = pd.read_csv("all_currencies_with_indicators_updated.csv")
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Filter to last 12 months for validation
            twelve_months_ago = df['datetime'].max() - timedelta(days=365)
            df = df[df['datetime'] >= twelve_months_ago].copy()
            
            print(f"✅ Loaded data: {len(df)} rows, {df['pair'].nunique()} pairs (last 12 months)")
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
                        feature_cols: List[str], confidence_threshold: float = 0.003) -> Dict:
        """Generate trading signals with optimized parameters."""
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
    
    def run_validation_test(self, data: pd.DataFrame, model, scaler: StandardScaler, 
                          feature_cols: List[str]) -> Dict:
        """Run validation test with optimized parameters."""
        
        print("🚀 Running validation test with optimized parameters...")
        
        # Optimized parameters from our analysis
        optimized_params = {
            'confidence_threshold': 0.003,
            'stop_loss': 0.02,
            'take_profit': 0.06,
            'max_position_size': 0.1,
            'max_daily_trades': 3,
            'initial_capital': 10000,
            'slippage': 0.002,
            'transaction_cost': 0.001,
            'step_size': 5
        }
        
        # Initialize trading state
        initial_capital = optimized_params['initial_capital']
        current_capital = initial_capital
        positions = {}
        trade_history = []
        daily_trades = {}
        equity_curve = []
        
        # Process data chronologically
        total_steps = len(data)
        processed_steps = 0
        
        for i in range(30, len(data), optimized_params['step_size']):
            processed_steps += 1
            if processed_steps % 200 == 0:
                print(f"  Progress: {processed_steps}/{total_steps//optimized_params['step_size']} steps")
            
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
                    if pnl <= -optimized_params['stop_loss'] or pnl >= optimized_params['take_profit']:
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
                            'hold_time': (current_time - position['entry_time']).total_seconds() / 3600
                        })
                        
                        del positions[pair]
            
            # Check daily trade limit
            current_date = current_time.date()
            if current_date not in daily_trades:
                daily_trades[current_date] = 0
            
            if daily_trades[current_date] >= optimized_params['max_daily_trades']:
                continue
            
            # Generate signals if no position in current pair
            if current_pair not in positions:
                signals = self.generate_signals(
                    current_data, model, scaler, feature_cols, 
                    optimized_params['confidence_threshold']
                )
                
                if signals:
                    signal = signals['signal']
                    confidence = signals['confidence']
                    
                    # Calculate position size
                    position_size = min(
                        optimized_params['max_position_size'] * current_capital,
                        current_capital * 0.1
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
        
        print("✅ Validation test completed!")
        
        # Calculate performance metrics
        return self.calculate_performance_metrics(trade_history, equity_curve, initial_capital, optimized_params)
    
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
    
    def run_validation(self):
        """Run the complete validation test."""
        print("🚀 Starting Optimized Parameters Validation Test")
        print("=" * 60)
        
        # Load data and models
        model, scaler, data, feature_cols = self.load_data_and_models()
        if model is None or scaler is None or data is None:
            print("❌ Failed to load required data/models")
            return
        
        # Run validation test
        start_time = datetime.now()
        results = self.run_validation_test(data, model, scaler, feature_cols)
        end_time = datetime.now()
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 VALIDATION TEST RESULTS (12-Month Period)")
        print("=" * 60)
        print(f"⏱️  Test time: {end_time - start_time}")
        print(f"💰 Initial capital: $10,000")
        print(f"📈 Total return: {results['total_return_pct']:.2f}%")
        print(f"📊 Sharpe ratio: {results['sharpe_ratio']:.3f}")
        print(f"📉 Max drawdown: {results['max_drawdown']:.2%}")
        print(f"🎯 Win rate: {results['win_rate']:.1%}")
        print(f"🔄 Total trades: {results['total_trades']}")
        print(f"📊 Avg trade return: ${results['avg_trade_return']:.2f}")
        print(f"💹 Profit factor: {results['profit_factor']:.2f}")
        print(f"⏰ Avg hold time: {results['avg_hold_time']:.1f} hours")
        
        print(f"\n🔧 Optimized Parameters Used:")
        for key, value in results['params'].items():
            print(f"  {key}: {value}")
        
        # Save results
        if results['trade_history']:
            trades_df = pd.DataFrame(results['trade_history'])
            trades_df.to_csv('validation_test_results.csv', index=False)
            print(f"\n✅ Validation results saved to: validation_test_results.csv")
        
        # Create validation chart
        if results['equity_curve']:
            equity_df = pd.DataFrame(results['equity_curve'])
            equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])
            
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df['datetime'], equity_df['equity'])
            plt.title('Validation Test - Optimized Parameters Equity Curve (12 Months)')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('validation_test_equity.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Chart saved to: validation_test_equity.png")
        
        return results

def main():
    """Main function to run validation test."""
    tester = OptimizedParameterTester()
    results = tester.run_validation()
    
    if results:
        print("\n🎉 Validation test completed successfully!")
        if results['win_rate'] > 0.6 and results['sharpe_ratio'] > 0.1:
            print("✅ Optimized parameters show good performance on longer timeframe!")
        else:
            print("⚠️ Performance degraded on longer timeframe - may need further optimization")
    else:
        print("\n❌ Validation test failed!")

if __name__ == "__main__":
    main() 