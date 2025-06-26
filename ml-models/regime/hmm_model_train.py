import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import joblib
import os

# 📥 Load sample data (you can replace this later with real OHLCV from Twelve Data)
df = pd.read_csv('sample_ohlc.csv')

# 🧮 Feature engineering
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['volatility'] = df['log_return'].rolling(window=10).std()
df.dropna(inplace=True)

X = df[['log_return', 'volatility']].values

# 🤖 Train HMM
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
model.fit(X)

# 💾 Save model
os.makedirs('regime', exist_ok=True)
joblib.dump(model, 'regime/hmm_model.pkl')

print("✅ HMM model trained and saved as 'regime/hmm_model.pkl'")
