# Portfolio Optimization Pipeline

This pipeline implements Markowitz portfolio optimization for multi-currency trading strategies using your tuned LightGBM model.

## 🎯 Overview

The pipeline consists of three main components:

1. **Strategy Returns Extraction** (`extract_strategy_returns.py`)
2. **Markowitz Optimization** (`markowitz_optimizer.py`) 
3. **Portfolio Backtesting** (`backtest_portfolio.py`)

## 📁 Files

### Core Scripts
- `extract_strategy_returns.py` - Extracts per-currency strategy returns using tuned model signals
- `markowitz_optimizer.py` - Computes optimal portfolio weights (Min Variance, Max Sharpe, Efficient Frontier)
- `backtest_portfolio.py` - Backtests optimized portfolios against benchmarks
- `run_portfolio_pipeline.py` - Complete pipeline runner

### Input Data
- `enhanced_regime_features.csv` - Your enhanced dataset with technical indicators and regime features
- `lgbm_best_model.pkl` - Your tuned LightGBM model (74% F1 score)
- `feature_list_full_technical.txt` - List of features used by the model

### Output Files
- `strategy_returns.csv` - Matrix of strategy returns (datetime × currency pairs)
- `markowitz_results.csv` - Portfolio weights and performance metrics
- `markowitz_optimization.png` - Efficient frontier and optimization plots
- `portfolio_backtest_summary.csv` - Comprehensive backtest results
- `portfolio_backtest_results.png` - Performance comparison plots

## 🚀 Quick Start

### Option 1: Run Complete Pipeline
```bash
python run_portfolio_pipeline.py --data enhanced_regime_features.csv --output_dir portfolio_results
```

### Option 2: Run Individual Steps

#### Step 1: Extract Strategy Returns
```bash
python extract_strategy_returns.py \
    --data enhanced_regime_features.csv \
    --output strategy_returns.csv \
    --threshold 0.5 \
    --transaction_cost 0.0
```

#### Step 2: Markowitz Optimization
```bash
python markowitz_optimizer.py \
    --returns strategy_returns.csv \
    --output_dir . \
    --risk_free_rate 0.02
```

#### Step 3: Portfolio Backtest
```bash
python backtest_portfolio.py \
    --returns strategy_returns.csv \
    --weights markowitz_results.csv \
    --output_dir . \
    --rebalance_freq daily \
    --transaction_cost 0.0
```

## 📊 Strategy Returns Extraction

### What it does:
- Loads your tuned LightGBM model
- Generates trading signals for each currency pair
- Calculates strategy returns: `signal * true_return`
- Creates a matrix of aligned returns across all pairs

### Key Features:
- Uses your 74% F1 tuned model
- Supports transaction costs
- Handles missing data gracefully
- Outputs time-aligned returns matrix

### Output Format:
```csv
datetime,EURUSD,GBPUSD,AUDUSD,USDJPY,NZDUSD
2023-01-02,0.0012,-0.0004,0.0009,0.0003,-0.0001
2023-01-03,0.0007,0.0001,-0.0002,0.0008,0.0005
...
```

## 🎯 Markowitz Optimization

### What it computes:
- **Minimum Variance Portfolio**: Lowest risk allocation
- **Maximum Sharpe Portfolio**: Best risk-adjusted returns
- **Efficient Frontier**: Risk-return trade-off curve

### Optimization Features:
- Long-only constraints (no short selling)
- Risk-free rate consideration
- Annualized metrics (assuming 4-hour data)
- Correlation analysis

### Output Metrics:
- Portfolio weights per currency pair
- Annualized return, volatility, Sharpe ratio
- Maximum drawdown, VaR, profit factor
- Efficient frontier coordinates

## 📈 Portfolio Backtesting

### What it compares:
- Individual currency performance
- Minimum variance portfolio
- Maximum Sharpe portfolio
- Equal-weight benchmark

### Backtest Features:
- Multiple rebalancing frequencies (daily/weekly/monthly)
- Transaction cost modeling
- Rolling performance metrics
- Drawdown analysis

### Performance Metrics:
- Total and annualized returns
- Sharpe ratio and volatility
- Maximum drawdown
- Win rate and profit factor
- Calmar ratio

## 🔧 Configuration Options

### Signal Generation
- `--threshold`: Model prediction threshold (default: 0.5)
- `--transaction_cost`: Fixed transaction cost (default: 0.0)

### Portfolio Optimization
- `--risk_free_rate`: Annual risk-free rate (default: 0.02)
- `--output_dir`: Results directory

### Backtesting
- `--rebalance_freq`: Portfolio rebalancing frequency
- `--transaction_cost`: Trading costs

## 📊 Expected Results

Based on your tuned model performance (74% F1, 15x cumulative return), you should see:

### Individual Currencies
- Varying performance across pairs
- Different risk-return profiles
- Correlation patterns

### Optimized Portfolios
- **Min Variance**: Lower volatility than individual assets
- **Max Sharpe**: Higher risk-adjusted returns
- **Efficient Frontier**: Clear risk-return trade-off

### Backtest Comparison
- Portfolio diversification benefits
- Risk reduction through optimization
- Performance vs. equal-weight benchmark

## 🎯 Key Insights to Look For

1. **Diversification Benefits**: How much risk reduction from portfolio optimization?
2. **Currency Correlations**: Which pairs move together/opposite?
3. **Optimal Weights**: Which currencies get highest/lowest allocations?
4. **Performance Persistence**: How stable are the optimized portfolios?

## 🚨 Important Notes

### Data Requirements
- Ensure `enhanced_regime_features.csv` has all required features
- Model file `lgbm_best_model.pkl` must be present
- Feature list file must match model training features

### Assumptions
- 4-hour data frequency (6 observations per trading day)
- 252 trading days per year
- Long-only positions (no short selling)
- No leverage constraints

### Limitations
- Historical optimization (look-ahead bias possible)
- Assumes stable correlations
- No regime-switching in optimization
- Fixed rebalancing frequency

## 🔄 Next Steps

After running the pipeline:

1. **Analyze Results**: Review plots and metrics
2. **Parameter Tuning**: Adjust thresholds and costs
3. **Risk Management**: Implement position sizing
4. **Live Trading**: Consider implementation challenges
5. **Monitoring**: Track performance vs. backtest

## 📞 Troubleshooting

### Common Issues:
- **Missing model file**: Ensure `lgbm_best_model.pkl` exists
- **Feature mismatch**: Check `feature_list_full_technical.txt`
- **Memory errors**: Reduce dataset size or use sampling
- **Optimization failures**: Check data quality and constraints

### Debug Mode:
Add `--verbose` flag to individual scripts for detailed output.

---

**🎉 Ready to optimize your multi-currency trading strategy!** 