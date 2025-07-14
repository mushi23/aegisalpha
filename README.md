🛡️ AegisAlpha

AegisAlpha is an experimental, data-driven trading analytics platform under active development. It leverages machine learning, financial APIs, and strategy backtesting to help identify actionable signals and trading regimes.

⚠️ This project is currently under active development. Features are evolving rapidly, and architecture may change.

🔍 Overview

AegisAlpha combines:

📊 Market regime detection (e.g., HMM, GMM)

📈 OHLCV-based analytics and forecasting

🧠 Machine learning models (LSTM, XGBoost)

🧪 Backtesting of strategies

🌐 FastAPI for ML serving and Laravel for API routing

It aims to provide:

Quant insight generation

Real-time analytics using financial APIs

Flexible strategy development/testing

💠 Tech Stack

Layer

Tech

Backend API

Laravel PHP (v12)

ML Engine

Python 3.12 (FastAPI, scikit-learn, etc.)

Data Source

Twelve Data API

Infra

GitHub, Conda, Uvicorn

🚧 Roadmap

Phase

Milestone

Status

1

Laravel backend setup with routing and basic API endpoint

✅ Complete

2

Twelve Data OHLCV integration + volatility/return calculation

✅ Complete

3

LSTM and XGBoost prediction models for 5s/1m forecasts

⏳ In progress

4

Regime detection (HMM + GMM) analytics layer

⏳ In progress

5

Add backtesting engine for strategy simulation

⏳ Pending

6

Frontend dashboard (e.g., React/Vue)

⏳ Pending

7

Docker + Deployment CI/CD setup

⏳ Pending

8

User authentication + secure API key management

⏳ Pending

⚙️ Setup Guide

1. Clone the Repository

git clone https://github.com/mushi23/aegisalpha.git
cd aegisalpha

2. Setup Laravel

cp .env.example .env
php artisan key:generate
php artisan serve

Set your Twelve Data API key in .env:

TWELVE_DATA_API_KEY=your_api_key_here

3. Run the Python ML Server

# (inside python/ or root if unified)
conda activate aegisalpha-env  # or use your virtualenv
uvicorn main:app --reload --port 8001

🔌 API Endpoints

GET /api/regime – returns detected regime index (HMM/GMM)

GET /api/ohlcv – returns OHLCV log return and volatility

POST /api/predict – feeds historical data to LSTM/XGBoost

🧪 Sample Output

{
  "log_return": -0.0034,
  "volatility": 0.0034
}

📁 Folder Structure

The repository is organized as follows:

- `app/` – Laravel backend (controllers, models, services)
- `ml-models/` – Python ML engine (notebooks, scripts, models, data processing)
- `public/` – Public assets for the web frontend
- `resources/` – Vue/TS frontend components and layouts
- `config/` – Laravel configuration files
- `database/` – Migrations, factories, and seeders
- `routes/` – Laravel route definitions
- `tests/` – Feature and unit tests (PHP)
- `requirements.txt`, `environment.yml` – Python dependencies
- `package.json` – Node dependencies for frontend
- `README.md` – Project documentation

📝 To Do (Expanded)

- [x] Laravel backend setup
- [x] Twelve Data API integration
- [x] Basic ML model integration (LSTM, XGBoost)
- [ ] Regime detection analytics (HMM, GMM)
- [ ] Backtesting engine for strategy simulation
- [ ] Frontend dashboard (Vue/TS)
- [ ] Dockerization and CI/CD
- [ ] User authentication and API key management
- [ ] Enhanced error handling and logging
- [ ] Automated tests for ML and API endpoints
- [ ] Documentation for contributing and usage
- [ ] Example Jupyter notebooks for research
- [ ] Data pipeline for new market data

💡 Usage

- For research: Use the provided Jupyter notebooks and scripts in `ml-models/` to experiment with new models, features, and strategies.
- For practical trading: Integrate the API endpoints with your trading infrastructure. **Note:** This project is for research and educational purposes. Use with caution in live trading environments.

🤝 Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or documentation improvements. See the folder structure above to find where your changes might fit best.

⚠️ Disclaimer (Expanded)

This repository is for educational and research purposes only. No part of this codebase constitutes financial advice. Trading financial instruments involves significant risk. The authors and contributors are not responsible for any financial losses incurred through the use of this software.

👨‍💼 Author

Mamuch Dak
GitHub: @mushi23

🧪 This repo is for educational and research purposes. Use it with caution in live trading environments.
