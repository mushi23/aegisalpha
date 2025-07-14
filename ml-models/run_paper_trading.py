#!/usr/bin/env python3
"""
Paper Trading Launcher
Simple script to run paper trading simulation with configurable parameters.
"""

import json
import argparse
from live_trading_system import LiveTradingSystem
import pandas as pd

def load_config(config_file: str = "live_trading_config.json"):
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded configuration from {config_file}")
        return config
    except Exception as e:
        print(f"❌ Could not load config: {e}")
        return None

def run_paper_trading_simulation(config: dict, data_file: str = None):
    """Run paper trading simulation with given configuration."""
    
    # Initialize trading system with config
    trading_config = config['trading']
    trading_system = LiveTradingSystem(
        initial_capital=trading_config['initial_capital'],
        max_position_size=trading_config['max_position_size'],
        max_daily_trades=trading_config['max_daily_trades'],
        slippage=trading_config['slippage'],
        transaction_cost=trading_config['transaction_cost'],
        confidence_threshold=trading_config['confidence_threshold'],
        stop_loss=trading_config['stop_loss'],
        take_profit=trading_config['take_profit']
    )
    
    # Load market data
    if data_file is None:
        data_file = config['data_source']['file_path']
    
    try:
        market_data = pd.read_csv(data_file)
        market_data['datetime'] = pd.to_datetime(market_data['datetime'])
        print(f"✅ Loaded market data: {len(market_data)} rows from {data_file}")
    except Exception as e:
        print(f"❌ Could not load market data: {e}")
        return
    
    # Filter out weekends and non-trading hours
    market_data['day_of_week'] = market_data['datetime'].dt.dayofweek
    market_data['hour'] = market_data['datetime'].dt.hour
    
    # Keep only weekdays (Monday=0 to Friday=4) and reasonable hours (0-23 for forex)
    trading_data = market_data[
        (market_data['day_of_week'] < 5) &  # Monday to Friday
        (market_data['hour'] >= 0) & (market_data['hour'] <= 23)  # 24-hour forex market
    ].copy()
    
    print(f"✅ Filtered to trading hours: {len(trading_data)} rows")
    
    # Load models
    if not trading_system.load_models_and_data():
        print("❌ Failed to load models. Exiting.")
        return
    
    # Run simulation
    print("\n🚀 Starting paper trading simulation...")
    print("="*60)
    
    # Process data chronologically, but only every N steps to avoid overtrading
    step_size = 10  # Process every 10th data point to simulate realistic trading frequency
    
    for i in range(30, len(trading_data), step_size):
        current_data = trading_data.iloc[:i+1]
        current_price = current_data.iloc[-1]['close']
        current_time = current_data.iloc[-1]['datetime']
        current_pair = current_data.iloc[-1]['pair']
        
        # Print progress every 1000 iterations
        if i % 1000 == 0:
            print(f"⏰ Processing {current_time} - {current_pair} @ ${current_price:.4f}")
        
        # Check exit conditions for existing positions
        for pair in list(trading_system.positions.keys()):
            trading_system.check_exit_conditions(pair, current_price)
        
        # Generate new signals
        signals = trading_system.generate_live_signals(current_data)
        
        # Execute trades for current pair (only if we have signals and meet risk limits)
        if signals:
            for signal_type, signal in signals.items():
                if trading_system.check_risk_limits(current_pair, signal):
                    trading_system.execute_trade(current_pair, signal, current_price)
                    break  # Only execute one trade per iteration
        
        # Update equity curve
        trading_system.update_equity_curve()
        
        # Print summary every 5000 iterations
        if i % 5000 == 0:
            trading_system.print_performance_summary()
    
    # Final performance summary
    print("\n" + "="*60)
    print("🏁 PAPER TRADING SIMULATION COMPLETED")
    print("="*60)
    trading_system.print_performance_summary()
    
    # Save results
    trading_system.save_trading_data("paper_trading_results")
    
    return trading_system

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run paper trading simulation")
    parser.add_argument("--config", type=str, default="live_trading_config.json",
                       help="Configuration file path")
    parser.add_argument("--data", type=str, default=None,
                       help="Market data file path (overrides config)")
    parser.add_argument("--capital", type=float, default=None,
                       help="Initial capital (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        return
    
    # Override capital if specified
    if args.capital:
        config['trading']['initial_capital'] = args.capital
        print(f"💰 Using initial capital: ${args.capital:,.2f}")
    
    # Run simulation
    trading_system = run_paper_trading_simulation(config, args.data)
    
    if trading_system:
        print("\n🎉 Paper trading simulation completed successfully!")
        print("📁 Results saved to paper_trading_results_*")
    else:
        print("\n❌ Paper trading simulation failed!")

if __name__ == "__main__":
    main() 