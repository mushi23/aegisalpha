#!/usr/bin/env python3
"""
Compounding Trading Simulation
Simulates the strategy with compounding returns and more frequent trading over a full year
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class CompoundingSimulation:
    def __init__(self):
        self.results = {}
        
    def load_data_and_models(self):
        """Load data and models for simulation."""
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
        
        # Load data (use full dataset for year simulation)
        try:
            df = pd.read_csv("all_currencies_with_indicators_updated.csv")
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Filter to last 2 years for simulation
            two_years_ago = df['datetime'].max() - timedelta(days=730)
            df = df[df['datetime'] >= two_years_ago].copy()
            
            print(f"✅ Loaded data: {len(df)} rows, {df['pair'].nunique()} pairs (last 2 years)")
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
    
    def run_compounding_simulation(self, data: pd.DataFrame, model, scaler: StandardScaler, 
                                 feature_cols: List[str], config: Dict) -> Dict:
        """Run compounding simulation with different trading frequencies."""
        
        print(f"🚀 Running compounding simulation with {config['max_daily_trades']} max daily trades...")
        
        # Initialize trading state
        initial_capital = config['initial_capital']
        current_capital = initial_capital
        positions = {}
        trade_history = []
        daily_trades = {}
        equity_curve = []
        monthly_returns = []
        
        # Configuration parameters
        max_position_size = config['max_position_size']
        max_daily_trades = config['max_daily_trades']
        slippage = config['slippage']
        transaction_cost = config['transaction_cost']
        confidence_threshold = config['confidence_threshold']
        stop_loss = config['stop_loss']
        take_profit = config['take_profit']
        step_size = config.get('step_size', 3)  # More frequent trading
        
        # Process data chronologically
        total_steps = len(data)
        processed_steps = 0
        
        for i in range(30, len(data), step_size):
            processed_steps += 1
            if processed_steps % 500 == 0:
                print(f"  Progress: {processed_steps}/{total_steps//step_size} steps")
            
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
                            'hold_time': (current_time - position['entry_time']).total_seconds() / 3600,
                            'capital_at_entry': position['capital_at_entry'],
                            'capital_at_exit': current_capital
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
                    
                    # Calculate position size (compounding)
                    position_size = min(
                        max_position_size * current_capital,  # Compounding position sizing
                        current_capital * 0.15  # Max 15% per trade for compounding
                    )
                    
                    # Open position
                    direction = 'long' if signal == 1 else 'short'
                    positions[current_pair] = {
                        'direction': direction,
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'size': position_size,
                        'confidence': confidence,
                        'capital_at_entry': current_capital
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
                'capital': current_capital,
                'unrealized_pnl': total_equity - current_capital
            })
        
        print("✅ Compounding simulation completed!")
        
        # Calculate performance metrics
        return self.calculate_compounding_metrics(trade_history, equity_curve, initial_capital, config)
    
    def calculate_compounding_metrics(self, trade_history: List, equity_curve: List, 
                                    initial_capital: float, config: Dict) -> Dict:
        """Calculate comprehensive compounding performance metrics."""
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
                'compounded_return': 0,
                'annualized_return': 0,
                'params': config
            }
        
        # Basic metrics
        total_trades = len(trade_history)
        winning_trades = len([t for t in trade_history if t['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Returns
        total_pnl = sum(t['pnl'] for t in trade_history)
        total_return = total_pnl / initial_capital
        total_return_pct = total_return * 100
        
        # Compounded return (final capital / initial capital)
        if equity_curve:
            final_capital = equity_curve[-1]['equity']
            compounded_return = (final_capital / initial_capital - 1) * 100
        else:
            compounded_return = total_return_pct
        
        # Annualized return (assuming 1 year)
        annualized_return = ((final_capital / initial_capital) ** (1/1) - 1) * 100
        
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
            'compounded_return': compounded_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_trade_return': avg_trade_return,
            'profit_factor': profit_factor,
            'avg_hold_time': avg_hold_time,
            'final_capital': final_capital if equity_curve else initial_capital,
            'trade_history': trade_history,
            'equity_curve': equity_curve,
            'params': config
        }
    
    def run_multiple_scenarios(self, data: pd.DataFrame, model, scaler: StandardScaler, 
                             feature_cols: List[str]) -> Dict:
        """Run multiple scenarios with different trading frequencies."""
        print("🔍 Running multiple compounding scenarios...")
        
        scenarios = {
            'Conservative': {
                'max_daily_trades': 2,
                'max_position_size': 0.08,
                'confidence_threshold': 0.003,
                'step_size': 5
            },
            'Moderate': {
                'max_daily_trades': 3,
                'max_position_size': 0.1,
                'confidence_threshold': 0.003,
                'step_size': 3
            },
            'Aggressive': {
                'max_daily_trades': 5,
                'max_position_size': 0.12,
                'confidence_threshold': 0.002,
                'step_size': 2
            },
            'High_Frequency': {
                'max_daily_trades': 8,
                'max_position_size': 0.15,
                'confidence_threshold': 0.002,
                'step_size': 1
            }
        }
        
        results = {}
        
        for scenario_name, config in scenarios.items():
            print(f"\n📊 Running {scenario_name} scenario...")
            
            # Add common parameters
            config.update({
                'initial_capital': 10000,
                'slippage': 0.002,
                'transaction_cost': 0.001,
                'stop_loss': 0.02,
                'take_profit': 0.06
            })
            
            # Run simulation
            result = self.run_compounding_simulation(data, model, scaler, feature_cols, config)
            results[scenario_name] = result
        
        return results
    
    def create_compounding_visualizations(self, results: Dict):
        """Create visualizations for compounding scenarios."""
        print("\n📊 Creating compounding visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Compounding Trading Scenarios - 1 Year Simulation', fontsize=16)
        
        # 1. Final capital comparison
        scenarios = list(results.keys())
        final_capitals = [results[s]['final_capital'] for s in scenarios]
        
        bars = axes[0, 0].bar(scenarios, final_capitals, color=['lightblue', 'lightgreen', 'orange', 'red'], alpha=0.7)
        axes[0, 0].set_ylabel('Final Capital ($)')
        axes[0, 0].set_title('Final Capital by Scenario')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, capital in zip(bars, final_capitals):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 100,
                           f'${capital:,.0f}', ha='center', va='bottom')
        
        # 2. Annualized returns
        annualized_returns = [results[s]['annualized_return'] for s in scenarios]
        
        bars = axes[0, 1].bar(scenarios, annualized_returns, color=['lightblue', 'lightgreen', 'orange', 'red'], alpha=0.7)
        axes[0, 1].set_ylabel('Annualized Return (%)')
        axes[0, 1].set_title('Annualized Returns by Scenario')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ret in zip(bars, annualized_returns):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{ret:.1f}%', ha='center', va='bottom')
        
        # 3. Sharpe ratios
        sharpe_ratios = [results[s]['sharpe_ratio'] for s in scenarios]
        
        bars = axes[0, 2].bar(scenarios, sharpe_ratios, color=['lightblue', 'lightgreen', 'orange', 'red'], alpha=0.7)
        axes[0, 2].set_ylabel('Sharpe Ratio')
        axes[0, 2].set_title('Risk-Adjusted Returns')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, sharpe in zip(bars, sharpe_ratios):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{sharpe:.3f}', ha='center', va='bottom')
        
        # 4. Win rates
        win_rates = [results[s]['win_rate'] * 100 for s in scenarios]
        
        bars = axes[1, 0].bar(scenarios, win_rates, color=['lightblue', 'lightgreen', 'orange', 'red'], alpha=0.7)
        axes[1, 0].set_ylabel('Win Rate (%)')
        axes[1, 0].set_title('Win Rates by Scenario')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{rate:.1f}%', ha='center', va='bottom')
        
        # 5. Max drawdowns
        max_drawdowns = [results[s]['max_drawdown'] * 100 for s in scenarios]
        
        bars = axes[1, 1].bar(scenarios, max_drawdowns, color=['lightblue', 'lightgreen', 'orange', 'red'], alpha=0.7)
        axes[1, 1].set_ylabel('Max Drawdown (%)')
        axes[1, 1].set_title('Risk Metrics')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, dd in zip(bars, max_drawdowns):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{dd:.1f}%', ha='center', va='bottom')
        
        # 6. Equity curves comparison
        for i, (scenario, result) in enumerate(results.items()):
            if result['equity_curve']:
                equity_df = pd.DataFrame(result['equity_curve'])
                equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])
                
                colors = ['blue', 'green', 'orange', 'red']
                axes[1, 2].plot(equity_df['datetime'], equity_df['equity'], 
                               label=scenario, color=colors[i], linewidth=2)
        
        axes[1, 2].set_xlabel('Date')
        axes[1, 2].set_ylabel('Equity ($)')
        axes[1, 2].set_title('Equity Curves Comparison')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('compounding_scenarios_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Compounding analysis charts saved to: compounding_scenarios_analysis.png")
    
    def run_compounding_analysis(self):
        """Run the complete compounding analysis."""
        print("🚀 Starting Compounding Trading Simulation")
        print("=" * 60)
        
        # Load data and models
        model, scaler, data, feature_cols = self.load_data_and_models()
        if model is None or scaler is None or data is None:
            print("❌ Failed to load required data/models")
            return
        
        # Run multiple scenarios
        start_time = datetime.now()
        results = self.run_multiple_scenarios(data, model, scaler, feature_cols)
        end_time = datetime.now()
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 COMPOUNDING SIMULATION RESULTS")
        print("=" * 60)
        print(f"⏱️  Simulation time: {end_time - start_time}")
        
        for scenario_name, result in results.items():
            print(f"\n📈 {scenario_name.upper()} SCENARIO:")
            print("-" * 40)
            print(f"💰 Initial Capital: $10,000")
            print(f"💰 Final Capital: ${result['final_capital']:,.2f}")
            print(f"📈 Total Return: {result['total_return_pct']:.2f}%")
            print(f"📈 Compounded Return: {result['compounded_return']:.2f}%")
            print(f"📈 Annualized Return: {result['annualized_return']:.2f}%")
            print(f"📊 Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print(f"📉 Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"🎯 Win Rate: {result['win_rate']:.1%}")
            print(f"🔄 Total Trades: {result['total_trades']}")
            print(f"📊 Avg Trade Return: ${result['avg_trade_return']:.2f}")
            print(f"💹 Profit Factor: {result['profit_factor']:.2f}")
            print(f"⏰ Avg Hold Time: {result['avg_hold_time']:.1f} hours")
            
            # Save individual scenario results
            if result['trade_history']:
                trades_df = pd.DataFrame(result['trade_history'])
                trades_df.to_csv(f'compounding_{scenario_name.lower()}_results.csv', index=False)
                print(f"✅ {scenario_name} results saved to: compounding_{scenario_name.lower()}_results.csv")
        
        # Create visualizations
        self.create_compounding_visualizations(results)
        
        # Find best scenario
        best_scenario = max(results.keys(), key=lambda x: results[x]['annualized_return'])
        print(f"\n🏆 BEST SCENARIO: {best_scenario}")
        print(f"   Annualized Return: {results[best_scenario]['annualized_return']:.2f}%")
        print(f"   Final Capital: ${results[best_scenario]['final_capital']:,.2f}")
        
        return results

def main():
    """Main function to run compounding analysis."""
    simulator = CompoundingSimulation()
    results = simulator.run_compounding_analysis()
    
    if results:
        print("\n🎉 Compounding simulation completed successfully!")
    else:
        print("\n❌ Compounding simulation failed!")

if __name__ == "__main__":
    main() 