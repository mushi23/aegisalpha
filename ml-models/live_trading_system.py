#!/usr/bin/env python3
"""
Live Trading System with Paper Trading
Real-time signal generation, risk management, and performance tracking.
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
warnings.filterwarnings('ignore')

class LiveTradingSystem:
    def __init__(self, 
                 initial_capital: float = 100000,
                 max_position_size: float = 0.1,
                 max_daily_trades: int = 5,
                 slippage: float = 0.002,
                 transaction_cost: float = 0.001,
                 confidence_threshold: float = 0.002,
                 stop_loss: float = 0.02,
                 take_profit: float = 0.03):
        """
        Initialize live trading system
        
        Args:
            initial_capital: Starting capital for paper trading
            max_position_size: Maximum position size as fraction of capital
            max_daily_trades: Maximum trades per day per currency pair
            slippage: Slippage per trade
            transaction_cost: Transaction cost per trade
            confidence_threshold: Minimum confidence for trade execution
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_daily_trades = max_daily_trades
        self.slippage = slippage
        self.transaction_cost = transaction_cost
        self.confidence_threshold = confidence_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Trading state
        self.positions = {}  # Current open positions
        self.trade_history = []  # All completed trades
        self.daily_trades = {}  # Daily trade count per pair
        self.performance_metrics = {}
        
        # Models and data
        self.models = {}
        self.scaler = None
        self.feature_cols = []
        
        # Performance tracking
        self.equity_curve = []
        self.signal_history = []
        
        print(f"🚀 Live Trading System initialized with ${initial_capital:,.2f} capital")
    
    def load_models_and_data(self):
        """Load trained models and prepare for live trading."""
        print("🔄 Loading models and data for live trading...")
        
        # Load models
        try:
            self.models['xgb_regression'] = joblib.load('models_future_return_5_regression/xgboost_future_return_5_regression.pkl')
            self.models['xgb_binary'] = joblib.load('models_target_binary_classification/xgboost_target_binary_classification.pkl')
            print("✅ Loaded XGBoost models")
        except Exception as e:
            print(f"❌ Could not load models: {e}")
            return False
        
        # Load scaler
        try:
            self.scaler = joblib.load('feature_scaler.pkl')
            print("✅ Loaded feature scaler")
        except Exception as e:
            print(f"❌ Could not load scaler: {e}")
            return False
        
        # Feature columns
        self.feature_cols = [
            'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_mid', 'support', 'resistance'
        ]
        
        print("✅ Live trading system ready")
        return True
    
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
        
        # Support and Resistance (simplified)
        df['support'] = df['close'].rolling(window=20).min()
        df['resistance'] = df['close'].rolling(window=20).max()
        
        # Volatility
        df['volatility_5'] = df['close'].pct_change().rolling(window=5).std()
        
        return df
    
    def generate_live_signals(self, market_data: pd.DataFrame) -> Dict:
        """Generate trading signals for live data."""
        if market_data.empty or len(market_data) < 30:
            return {}
        
        # Calculate features
        df = self.calculate_technical_features(market_data)
        
        # Get latest data point
        latest_data = df.iloc[-1:]
        
        # Check for required features
        missing_features = [f for f in self.feature_cols if f not in latest_data.columns]
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            return {}
        
        # Prepare features for prediction
        X = latest_data[self.feature_cols].values
        
        # Check for NaN values
        if np.isnan(X).any():
            print("⚠️ NaN values in features, skipping signal generation")
            return {}
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        signals = {}
        
        # Generate regression signals
        if 'xgb_regression' in self.models:
            xgb_pred = self.models['xgb_regression'].predict(X_scaled)[0]
            confidence = abs(xgb_pred)
            
            if confidence > self.confidence_threshold:
                signal = 1 if xgb_pred > 0 else -1
                signals['xgb_regression'] = {
                    'signal': signal,
                    'confidence': confidence,
                    'predicted_return': xgb_pred,
                    'timestamp': datetime.now()
                }
        
        # Generate binary signals
        if 'xgb_binary' in self.models:
            xgb_bin_prob = self.models['xgb_binary'].predict_proba(X_scaled)[0, 1]
            confidence = max(xgb_bin_prob, 1 - xgb_bin_prob)
            
            if xgb_bin_prob > 0.6 and confidence > self.confidence_threshold:
                signals['xgb_binary'] = {
                    'signal': 1,
                    'confidence': confidence,
                    'probability': xgb_bin_prob,
                    'timestamp': datetime.now()
                }
        
        return signals
    
    def check_risk_limits(self, pair: str, signal: Dict) -> bool:
        """Check if trade meets risk management criteria."""
        # Check daily trade limit
        today = datetime.now().date()
        if today not in self.daily_trades:
            self.daily_trades[today] = {}
        
        if pair not in self.daily_trades[today]:
            self.daily_trades[today][pair] = 0
        
        if self.daily_trades[today][pair] >= self.max_daily_trades:
            print(f"⚠️ Daily trade limit reached for {pair}")
            return False
        
        # Check if we already have a position in this pair
        if pair in self.positions:
            print(f"⚠️ Already have position in {pair}")
            return False
        
        # Check capital adequacy
        position_size = self.current_capital * self.max_position_size
        if position_size > self.current_capital * 0.8:  # Don't use more than 80% of capital
            print(f"⚠️ Insufficient capital for new position")
            return False
        
        return True
    
    def execute_trade(self, pair: str, signal: Dict, current_price: float) -> bool:
        """Execute a trade in the paper trading environment."""
        if not self.check_risk_limits(pair, signal):
            return False
        
        # Calculate position size
        base_position_size = self.current_capital * self.max_position_size
        confidence = signal.get('confidence', 0.5)
        position_size = base_position_size * confidence
        
        # Calculate trade value
        trade_value = position_size
        trade_quantity = trade_value / current_price
        
        # Calculate costs
        total_cost = (self.slippage + self.transaction_cost) * trade_value
        
        # Execute trade
        trade = {
            'id': len(self.trade_history) + 1,
            'pair': pair,
            'side': 'buy' if signal['signal'] > 0 else 'sell',
            'quantity': trade_quantity,
            'price': current_price,
            'value': trade_value,
            'costs': total_cost,
            'timestamp': datetime.now(),
            'signal_type': list(signal.keys())[0] if signal else 'unknown',
            'confidence': confidence
        }
        
        # Update positions
        self.positions[pair] = {
            'side': trade['side'],
            'quantity': trade_quantity,
            'entry_price': current_price,
            'entry_time': datetime.now(),
            'stop_loss': current_price * (1 - self.stop_loss) if signal['signal'] > 0 else current_price * (1 + self.stop_loss),
            'take_profit': current_price * (1 + self.take_profit) if signal['signal'] > 0 else current_price * (1 - self.take_profit)
        }
        
        # Update capital
        self.current_capital -= total_cost
        
        # Update daily trade count
        today = datetime.now().date()
        if today not in self.daily_trades:
            self.daily_trades[today] = {}
        if pair not in self.daily_trades[today]:
            self.daily_trades[today][pair] = 0
        self.daily_trades[today][pair] += 1
        
        # Add to trade history
        self.trade_history.append(trade)
        
        print(f"✅ Executed {trade['side']} trade for {pair}: {trade_quantity:.4f} @ ${current_price:.4f}")
        print(f"   Value: ${trade_value:.2f}, Costs: ${total_cost:.2f}")
        
        return True
    
    def check_exit_conditions(self, pair: str, current_price: float) -> bool:
        """Check if position should be closed."""
        if pair not in self.positions:
            return False
        
        position = self.positions[pair]
        
        # Check stop loss
        if position['side'] == 'buy':
            if current_price <= position['stop_loss']:
                return self.close_position(pair, current_price, 'stop_loss')
            elif current_price >= position['take_profit']:
                return self.close_position(pair, current_price, 'take_profit')
        else:  # sell
            if current_price >= position['stop_loss']:
                return self.close_position(pair, current_price, 'stop_loss')
            elif current_price <= position['take_profit']:
                return self.close_position(pair, current_price, 'take_profit')
        
        return False
    
    def close_position(self, pair: str, current_price: float, reason: str) -> bool:
        """Close a position and calculate P&L."""
        if pair not in self.positions:
            return False
        
        position = self.positions[pair]
        
        # Calculate P&L
        if position['side'] == 'buy':
            pnl = (current_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - current_price) * position['quantity']
        
        # Calculate costs
        trade_value = position['quantity'] * current_price
        costs = (self.slippage + self.transaction_cost) * trade_value
        
        # Net P&L
        net_pnl = pnl - costs
        
        # Update capital
        self.current_capital += net_pnl
        
        # Record trade
        trade = {
            'id': len(self.trade_history) + 1,
            'pair': pair,
            'side': 'sell' if position['side'] == 'buy' else 'buy',
            'quantity': position['quantity'],
            'price': current_price,
            'value': trade_value,
            'costs': costs,
            'pnl': net_pnl,
            'timestamp': datetime.now(),
            'exit_reason': reason
        }
        
        self.trade_history.append(trade)
        
        # Remove position
        del self.positions[pair]
        
        print(f"🔚 Closed {pair} position: {reason}")
        print(f"   P&L: ${net_pnl:.2f}, Capital: ${self.current_capital:.2f}")
        
        return True
    
    def update_equity_curve(self):
        """Update equity curve with current performance."""
        total_pnl = sum([trade.get('pnl', 0) for trade in self.trade_history])
        equity = self.initial_capital + total_pnl
        
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': equity,
            'capital': self.current_capital,
            'open_positions': len(self.positions),
            'total_trades': len(self.trade_history)
        })
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trade_history:
            return {}
        
        # Filter completed trades (with P&L)
        completed_trades = [t for t in self.trade_history if 'pnl' in t]
        
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
        returns = [t['pnl'] / self.initial_capital for t in completed_trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Drawdown calculation
        equity_values = [e['equity'] for e in self.equity_curve]
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
        
        metrics = {
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
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'open_positions': len(self.positions)
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def print_performance_summary(self):
        """Print current performance summary."""
        metrics = self.calculate_performance_metrics()
        
        if not metrics:
            print("📊 No completed trades yet")
            return
        
        print("\n" + "="*60)
        print("📊 LIVE TRADING PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"💰 Capital: ${self.current_capital:,.2f} (Initial: ${self.initial_capital:,.2f})")
        print(f"📈 Total Return: {metrics['total_return']:.2%}")
        print(f"📊 Total P&L: ${metrics['total_pnl']:,.2f}")
        print(f"🎯 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"🏆 Win Rate: {metrics['win_rate']:.2%}")
        print(f"📊 Total Trades: {metrics['total_trades']}")
        print(f"✅ Winning Trades: {metrics['winning_trades']}")
        print(f"❌ Losing Trades: {metrics['losing_trades']}")
        print(f"💵 Avg Win: ${metrics['avg_win']:.2f}")
        print(f"💸 Avg Loss: ${metrics['avg_loss']:.2f}")
        print(f"📈 Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"🔓 Open Positions: {metrics['open_positions']}")
        
        if self.positions:
            print(f"\n🔓 Open Positions:")
            for pair, pos in self.positions.items():
                print(f"   {pair}: {pos['side']} {pos['quantity']:.4f} @ ${pos['entry_price']:.4f}")
    
    def save_trading_data(self, filename: str = None):
        """Save trading data to files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"live_trading_data_{timestamp}"
        
        # Save trade history
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            trades_df.to_csv(f"{filename}_trades.csv", index=False)
        
        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(f"{filename}_equity.csv", index=False)
        
        # Save performance metrics
        if self.performance_metrics:
            with open(f"{filename}_metrics.json", 'w') as f:
                json.dump(self.performance_metrics, f, indent=2, default=str)
        
        print(f"✅ Trading data saved to {filename}_*")
    
    def run_paper_trading_simulation(self, market_data: pd.DataFrame, update_interval: int = 60):
        """Run paper trading simulation with historical data."""
        print("🚀 Starting paper trading simulation...")
        
        if not self.load_models_and_data():
            print("❌ Failed to load models. Exiting.")
            return
        
        # Process data chronologically
        for i in range(30, len(market_data)):  # Start from 30 to have enough data for indicators
            current_data = market_data.iloc[:i+1]
            current_price = current_data.iloc[-1]['close']
            current_time = current_data.iloc[-1]['datetime']
            
            print(f"\n⏰ {current_time} - Price: ${current_price:.4f}")
            
            # Check exit conditions for existing positions
            for pair in list(self.positions.keys()):
                self.check_exit_conditions(pair, current_price)
            
            # Generate new signals
            signals = self.generate_live_signals(current_data)
            
            # Execute trades for each currency pair
            for pair in market_data['pair'].unique():
                pair_data = current_data[current_data['pair'] == pair]
                if not pair_data.empty:
                    pair_price = pair_data.iloc[-1]['close']
                    
                    # Check for signals for this pair
                    for signal_type, signal in signals.items():
                        if self.check_risk_limits(pair, signal):
                            self.execute_trade(pair, signal, pair_price)
            
            # Update equity curve
            self.update_equity_curve()
            
            # Print summary every 100 iterations
            if i % 100 == 0:
                self.print_performance_summary()
            
            # Simulate time delay
            time.sleep(0.1)  # 100ms delay for simulation
        
        # Final performance summary
        print("\n" + "="*60)
        print("🏁 PAPER TRADING SIMULATION COMPLETED")
        print("="*60)
        self.print_performance_summary()
        
        # Save results
        self.save_trading_data()

def main():
    """Main function to run paper trading simulation."""
    # Initialize trading system
    trading_system = LiveTradingSystem(
        initial_capital=100000,
        max_position_size=0.1,
        max_daily_trades=5,
        slippage=0.002,
        transaction_cost=0.001,
        confidence_threshold=0.002
    )
    
    # Load market data for simulation
    try:
        market_data = pd.read_csv("all_currencies_with_indicators_updated.csv")
        market_data['datetime'] = pd.to_datetime(market_data['datetime'])
        print(f"✅ Loaded market data: {len(market_data)} rows")
    except Exception as e:
        print(f"❌ Could not load market data: {e}")
        return
    
    # Run paper trading simulation
    trading_system.run_paper_trading_simulation(market_data)

if __name__ == "__main__":
    main() 