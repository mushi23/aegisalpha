#!/usr/bin/env python3
"""
Fast Paper Trading System
Simplified version focused on core trading functionality for quick testing
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

class FastPaperTrading:
    def __init__(self):
        self.results = {}
        
    def load_data_and_models(self):
        """Load data and models for fast trading."""
        print("🔄 Loading data and models...")
        
        # Load model (just use one for speed)
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
        
        # Load data (use smaller subset for speed)
        try:
            df = pd.read_csv("all_currencies_with_indicators_updated.csv")
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Filter to last 6 months for faster processing
            six_months_ago = df['datetime'].max() - timedelta(days=180)
            df = df[df['datetime'] >= six_months_ago].copy()
            
            print(f"✅ Loaded data: {len(df)} rows, {df['pair'].nunique()} pairs (last 6 months)")
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
    
    def run_fast_simulation(self, data: pd.DataFrame, model, scaler: StandardScaler, 
                          feature_cols: List[str]) -> Dict:
        """Run a fast trading simulation with optimized settings."""
        
        print("🚀 Starting fast paper trading simulation...")
        
        # Initialize trading state
        initial_capital = 10000
        current_capital = initial_capital
        positions = {}
        trade_history = []
        daily_trades = {}
        equity_curve = []
        
        # Optimized configuration for speed
        config = {
            'max_position_size': 0.1,  # 10% of capital
            'max_daily_trades': 3,
            'slippage': 0.002,
            'transaction_cost': 0.001,
            'confidence_threshold': 0.002,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'step_size': 10  # Process every 10th data point for speed
        }
        
        # Process data chronologically with larger step size
        total_steps = len(data)
        processed_steps = 0
        
        for i in range(30, len(data), config['step_size']):
            processed_steps += 1
            if processed_steps % 100 == 0:
                print(f"  Progress: {processed_steps}/{total_steps//config['step_size']} steps")
            
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
                    if pnl <= -config['stop_loss'] or pnl >= config['take_profit']:
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
                            'return_pct': pnl * 100
                        })
                        
                        del positions[pair]
            
            # Check daily trade limit
            current_date = current_time.date()
            if current_date not in daily_trades:
                daily_trades[current_date] = 0
            
            if daily_trades[current_date] >= config['max_daily_trades']:
                continue
            
            # Generate signals if no position in current pair
            if current_pair not in positions:
                signals = self.generate_signals(
                    current_data, model, scaler, feature_cols, 
                    config['confidence_threshold']
                )
                
                if signals:
                    signal = signals['signal']
                    confidence = signals['confidence']
                    
                    # Calculate position size
                    position_size = min(
                        config['max_position_size'] * current_capital,
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
        
        print("✅ Simulation completed!")
        
        # Calculate performance metrics
        return self.calculate_performance_metrics(trade_history, equity_curve, initial_capital)
    
    def calculate_performance_metrics(self, trade_history: List, equity_curve: List, initial_capital: float) -> Dict:
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
                'profit_factor': 0
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
            'trade_history': trade_history,
            'equity_curve': equity_curve
        }
    
    def run_fast_trading(self):
        """Run the complete fast trading system."""
        print("🚀 Starting Fast Paper Trading System")
        print("=" * 50)
        
        # Load data and models
        model, scaler, data, feature_cols = self.load_data_and_models()
        if model is None or scaler is None or data is None:
            print("❌ Failed to load required data/models")
            return
        
        # Run simulation
        start_time = datetime.now()
        results = self.run_fast_simulation(data, model, scaler, feature_cols)
        end_time = datetime.now()
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 FAST PAPER TRADING RESULTS")
        print("=" * 50)
        print(f"⏱️  Execution time: {end_time - start_time}")
        print(f"💰 Initial capital: $10,000")
        print(f"📈 Total return: {results['total_return_pct']:.2f}%")
        print(f"📊 Sharpe ratio: {results['sharpe_ratio']:.3f}")
        print(f"📉 Max drawdown: {results['max_drawdown']:.2%}")
        print(f"🎯 Win rate: {results['win_rate']:.1%}")
        print(f"🔄 Total trades: {results['total_trades']}")
        print(f"📊 Avg trade return: ${results['avg_trade_return']:.2f}")
        print(f"💹 Profit factor: {results['profit_factor']:.2f}")
        
        # Save results
        results_df = pd.DataFrame(results['trade_history'])
        if not results_df.empty:
            results_df.to_csv('fast_trading_results.csv', index=False)
            print(f"\n✅ Results saved to: fast_trading_results.csv")
        
        # Create simple visualization
        if results['equity_curve']:
            equity_df = pd.DataFrame(results['equity_curve'])
            equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])
            
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df['datetime'], equity_df['equity'])
            plt.title('Fast Paper Trading - Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('fast_trading_equity.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Chart saved to: fast_trading_equity.png")
        
        return results

def main():
    """Main function to run fast paper trading."""
    trader = FastPaperTrading()
    results = trader.run_fast_trading()
    
    if results:
        print("\n🎉 Fast paper trading completed successfully!")
    else:
        print("\n❌ Fast paper trading failed!")

if __name__ == "__main__":
    main() 