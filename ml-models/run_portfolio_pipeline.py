#!/usr/bin/env python3
"""
Portfolio Optimization Pipeline Runner
Executes the complete pipeline: extract returns → optimize → backtest
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run complete portfolio optimization pipeline")
    parser.add_argument("--data", type=str, default="enhanced_regime_features.csv",
                       help="Path to enhanced dataset")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Signal threshold for model predictions")
    parser.add_argument("--transaction_cost", type=float, default=0.0,
                       help="Transaction cost as decimal")
    parser.add_argument("--risk_free_rate", type=float, default=0.02,
                       help="Annual risk-free rate")
    parser.add_argument("--rebalance_freq", type=str, default="daily",
                       choices=['daily', 'weekly', 'monthly'],
                       help="Portfolio rebalancing frequency")
    parser.add_argument("--output_dir", type=str, default="portfolio_results",
                       help="Output directory for all results")
    parser.add_argument("--slippage", type=float, default=0.002,
                       help="Slippage per trade as decimal (e.g., 0.002 = 0.2%)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Starting Portfolio Optimization Pipeline")
    print("=" * 60)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Data source: {args.data}")
    print(f"🎯 Signal threshold: {args.threshold}")
    print(f"💰 Transaction cost: {args.transaction_cost}")
    print(f"📈 Risk-free rate: {args.risk_free_rate}")
    print(f"🔄 Rebalancing frequency: {args.rebalance_freq}")
    print(f"🔄 Slippage: {args.slippage}")
    print("=" * 60)
    
    # Step 1: Extract Strategy Returns
    returns_path = f"{args.output_dir}/strategy_returns.csv"
    step1_cmd = f"python extract_strategy_returns.py --data {args.data} --output {returns_path} --threshold {args.threshold} --transaction_cost {args.transaction_cost} --slippage {args.slippage}"
    
    if not run_command(step1_cmd, "Extracting strategy returns"):
        print("❌ Pipeline failed at step 1")
        sys.exit(1)
    
    # Step 2: Markowitz Optimization
    weights_path = f"{args.output_dir}/markowitz_results.csv"
    step2_cmd = f"python markowitz_optimizer.py --returns {returns_path} --output_dir {args.output_dir} --risk_free_rate {args.risk_free_rate}"
    
    if not run_command(step2_cmd, "Running Markowitz optimization"):
        print("❌ Pipeline failed at step 2")
        sys.exit(1)
    
    # Step 3: Portfolio Backtest
    step3_cmd = f"python backtest_portfolio.py --returns {returns_path} --weights {weights_path} --output_dir {args.output_dir} --rebalance_freq {args.rebalance_freq} --transaction_cost {args.transaction_cost} --slippage {args.slippage}"
    
    if not run_command(step3_cmd, "Running portfolio backtest"):
        print("❌ Pipeline failed at step 3")
        sys.exit(1)
    
    # Summary
    print("\n🎉 Portfolio Optimization Pipeline Completed Successfully!")
    print("=" * 60)
    print("📁 Generated Files:")
    print(f"   📊 Strategy Returns: {returns_path}")
    print(f"   🎯 Portfolio Weights: {weights_path}")
    print(f"   📈 Optimization Plots: {args.output_dir}/markowitz_optimization.png")
    print(f"   📊 Backtest Results: {args.output_dir}/portfolio_backtest_summary.csv")
    print(f"   📈 Backtest Plots: {args.output_dir}/portfolio_backtest_results.png")
    
    print("\n📋 Next Steps:")
    print("   1. Review the optimization plots to understand portfolio characteristics")
    print("   2. Analyze backtest results to compare portfolio performance")
    print("   3. Consider transaction costs and rebalancing frequency for live trading")
    print("   4. Implement risk management and position sizing")
    
    print(f"\n⏰ Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 