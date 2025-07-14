#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("all_currencies_combined.csv")
df.columns = df.columns.str.strip().str.lower()

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])

print(df['datetime'].head())
print(df.columns.tolist())

# Plot each currency pair separately
for pair in df['pair'].unique():
    sub_df = df[df['pair'] == pair]

    plt.figure(figsize=(12, 4))
    plt.plot(sub_df['datetime'], sub_df['close'], label=f'{pair} Close Price')
    plt.title(f'{pair} 4-Hour Close Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

df = pd.read_csv("all_currencies_combined.csv")
df.columns = df.columns.str.strip().str.lower()
df['datetime'] = pd.to_datetime(df['datetime'])

sequence_length = 60

X_lstm_all = []
y_lstm_all = []
xgb_frames = []

for pair in df['pair'].unique():
    pair_df = df[df['pair'] == pair].copy()
    pair_df.sort_values("datetime", inplace=True)

    # ===== LSTM Preprocessing =====
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(pair_df[['close']])
    
    X_lstm = []
    y_lstm = []

    for i in range(sequence_length, len(scaled_close)):
        X_lstm.append(scaled_close[i - sequence_length:i])
        y_lstm.append(scaled_close[i])  # or scaled_close[i, 0] for consistency

    X_lstm_all.extend(X_lstm)
    y_lstm_all.extend(y_lstm)

    # ===== XGBoost Features =====
    pair_df['return'] = pair_df['close'].pct_change()
    pair_df['lag_1'] = pair_df['close'].shift(1)
    pair_df['lag_2'] = pair_df['close'].shift(2)
    pair_df['lag_3'] = pair_df['close'].shift(3)
    pair_df['hour'] = pair_df['datetime'].dt.hour
    pair_df['minute'] = pair_df['datetime'].dt.minute
    pair_df['pair'] = pair  # ensure 'pair' column is preserved

    # === New: 5-bar forward return, log-return, and classification label ===
    pair_df['future_return_5'] = (pair_df['close'].shift(-5) - pair_df['close']) / pair_df['close']
    pair_df['future_log_return_5'] = np.log(pair_df['close'].shift(-5)) - np.log(pair_df['close'])
    threshold = 0.002
    pair_df['target_class'] = 0
    pair_df.loc[pair_df['future_return_5'] > threshold, 'target_class'] = 1
    pair_df.loc[pair_df['future_return_5'] < -threshold, 'target_class'] = -1

    pair_df.dropna(inplace=True)
    xgb_frames.append(pair_df)

    # Add after computing 'return' column
    cost_per_trade = 0.002 + 0.005  # 0.7% total
    if 'return' in pair_df.columns:
        pair_df['net_return'] = pair_df['return'] - cost_per_trade
        pair_df['label'] = (pair_df['net_return'] > 0).astype(int)

# === Save LSTM arrays ===
X_lstm_all = np.array(X_lstm_all)
y_lstm_all = np.array(y_lstm_all)

np.save("X_lstm_multicurrency.npy", X_lstm_all)
np.save("y_lstm_multicurrency.npy", y_lstm_all)

# === Save XGBoost-ready dataset ===
df_xgb_all = pd.concat(xgb_frames, ignore_index=True)
df_xgb_all.to_csv("multicurrency_xgb_ready.csv", index=False)

print(f"✅ Saved LSTM arrays: {X_lstm_all.shape}, {y_lstm_all.shape}")
print("✅ Saved XGBoost CSV: multicurrency_xgb_ready.csv")


# In[ ]:





# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# Load multi-currency cleaned dataset
df = pd.read_csv("all_currencies_combined.csv")
df.columns = df.columns.str.strip().str.lower()
df['datetime'] = pd.to_datetime(df['datetime'])

# Plot each currency pair separately
unique_pairs = df['pair'].unique()

for pair in unique_pairs:
    pair_df = df[df['pair'] == pair].sort_values('datetime')

    plt.figure(figsize=(14, 6))
    plt.plot(pair_df['datetime'], pair_df['close'], label=f'{pair} Close Price')
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.title(f"{pair} 4-Hour Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Load and parse datetime
df = pd.read_csv("eurusd_4hcleaned.csv")
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df['datetime'], df['close'], label='Close Price')
plt.xlabel("Time")
plt.ylabel("EUR/USD Close")
plt.title("EUR/USD 4-H Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[4]:


import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"

# Load combined OHLCV data
df = pd.read_csv("all_currencies_combined.csv")
df.columns = df.columns.str.strip().str.lower()
df['datetime'] = pd.to_datetime(df['datetime']) + pd.Timedelta(hours=3)

# Loop through each currency pair and plot candlestick chart
for pair in df['pair'].unique():
    pair_df = df[df['pair'] == pair].sort_values('datetime').tail(500)

    fig = go.Figure(data=[go.Candlestick(
        x=pair_df['datetime'],
        open=pair_df['open'],
        high=pair_df['high'],
        low=pair_df['low'],
        close=pair_df['close'],
        name=pair
    )])

    fig.update_layout(
        title=f"{pair} 4-Hour Candlestick Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600
    )

    fig.show()


# In[5]:


import pandas as pd
import ta

# Load combined dataset
df = pd.read_csv("all_currencies_combined.csv")
df.columns = df.columns.str.strip().str.lower()
df['datetime'] = pd.to_datetime(df['datetime'])

# Container for processed DataFrames
enriched_dfs = []

# Group by currency pair and apply indicators
for pair, group in df.groupby("pair"):
    group = group.sort_values("datetime").copy()

    # Technical indicators
    group['sma_20'] = ta.trend.sma_indicator(group['close'], window=20)
    group['ema_20'] = ta.trend.ema_indicator(group['close'], window=20)
    group['rsi'] = ta.momentum.rsi(group['close'], window=14)

    group['macd'] = ta.trend.macd(group['close'])
    group['macd_signal'] = ta.trend.macd_signal(group['close'])

    bb = ta.volatility.BollingerBands(close=group['close'], window=20)
    group['bb_upper'] = bb.bollinger_hband()
    group['bb_lower'] = bb.bollinger_lband()
    group['bb_mid'] = bb.bollinger_mavg()

    # Support & resistance
    group['support'] = group['low'].rolling(window=20).min()
    group['resistance'] = group['high'].rolling(window=20).max()

    # Drop rows with NaNs from indicators
    group.dropna(inplace=True)

    enriched_dfs.append(group)

# Combine all processed groups
df_enriched = pd.concat(enriched_dfs, ignore_index=True)

# Save
df_enriched.to_csv("all_currencies_with_indicators.csv", index=False)
print("✅ Enriched multi-currency dataset with indicators saved.")


# In[6]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("all_currencies_with_indicators.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(['pair', 'datetime'])

# Features to use
features = ['close', 'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal']
sequence_length = 60

# Containers
X_all, y_all = [], []

# Group by currency pair
for pair, group in df.groupby("pair"):
    group = group[features].dropna()
    
    # Scale features separately per pair
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(group)

    # Sequence generation
    for i in range(sequence_length, len(scaled)):
        X_all.append(scaled[i - sequence_length:i])
        y_all.append(scaled[i, 0])  # predict 'close'

# Final arrays
X_lstm = np.array(X_all)
y_lstm = np.array(y_all)

# Save to disk
np.save("X_lstm_with_indicators_multi.npy", X_lstm)
np.save("y_lstm_with_indicators_multi.npy", y_lstm)

print(f"✅ Saved LSTM-ready arrays with shape: X={X_lstm.shape}, y={y_lstm.shape}")


# In[7]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load multi-currency LSTM sequences
X = np.load("X_lstm_with_indicators_multi.npy")
y = np.load("y_lstm_with_indicators_multi.npy")
print(f"Data shapes: X={X.shape}, y={y.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Model architecture
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("LSTM Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Predict & evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"✅ LSTM Test MSE: {mse:.5f}")

# === Inverse transform utility ===
def inverse_transform(preds_1d, scaler, features):
    padded = np.concatenate([preds_1d.reshape(-1, 1), np.zeros((len(preds_1d), len(features)-1))], axis=1)
    return scaler.inverse_transform(padded)[:, 0]  # Only extract the inverse of 'close'

# For LSTM and XGBoost predictions, always inverse-transform both predictions and y_test before plotting or metrics

# Example for LSTM:
# lstm_preds_real = inverse_transform(lstm_preds, scaler, features)
# y_test_real = inverse_transform(y_test, scaler, features)
# plt.plot(y_test_real, label='Actual')
# plt.plot(lstm_preds_real, label='Predicted')

# Example for XGBoost:
# xgb_preds_real = inverse_transform(xgb_preds, scaler, features)
# y_test_real = inverse_transform(y_test, scaler, features)
# plt.plot(y_test_real, label='Actual')
# plt.plot(xgb_preds_real, label='Predicted')

# Plot predictions
plt.figure(figsize=(10, 5))
y_test_real = inverse_transform(y_test, scaler, features)
y_pred_real = inverse_transform(y_pred, scaler, features)
plt.plot(y_test_real, label='Actual')
plt.plot(y_pred_real, label='Predicted')
plt.title("LSTM: True vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save model & predictions
model.save("lstm_model_multi_currency.keras")
np.save("lstm_y_pred_multi.npy", y_pred)


# In[13]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Load and sort dataset
df = pd.read_csv("all_currencies_with_indicators.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(['pair', 'datetime'])

# Select features
features = ['close', 'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal']
df = df[features].dropna().reset_index(drop=True)

# Save unscaled 'close' values
real_close = df['close'].values

# Sequence generation (before scaling)
sequence_length = 60
X_raw, y_raw, y_real = [], [], []

for i in range(sequence_length, len(df)):
    seq = df.iloc[i - sequence_length:i].values  # shape: (60, 6)
    X_raw.append(seq)
    y_raw.append(df.iloc[i]['close'])            # for later scaling
    y_real.append(real_close[i])                 # true price

X_raw = np.array(X_raw)                          # (samples, 60, 6)
y_real = np.array(y_real)

# Flatten X for XGBoost: (samples, 360)
X_raw = X_raw.reshape(len(X_raw), -1)

# Train/test split on raw values (before scaling)
X_train_raw, X_test_raw, y_train_real, y_test_real = train_test_split(
    X_raw, y_real, test_size=0.2, shuffle=False
)

# Fit scaler on training only
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Now, scale y_train_real and y_test_real
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train_real.reshape(-1, 1)).ravel()
y_test = y_scaler.transform(y_test_real.reshape(-1, 1)).ravel()

# Train model
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)
joblib.dump(model, "xgboost_model_multicurrency.pkl")

# Predict
y_pred_scaled = model.predict(X_test)
y_pred_real = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Evaluate
mse_real = mean_squared_error(y_test_real, y_pred_real)
print(f"✅ Real XGBoost MSE: {mse_real:.5f}")

# Plot
plt.figure(figsize=(12, 6))
y_test_real = inverse_transform(y_test, y_scaler, features)
y_pred_real = inverse_transform(y_pred_real, y_scaler, features)
plt.plot(y_test_real, label='True Close Price')
plt.plot(y_pred_real, label='XGBoost Predicted')
plt.title("📈 XGBoost Predictions (Real Price Scale)")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[9]:


import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load multi-currency models
model_lstm = load_model('lstm_model_multi_currency.keras', compile=False)
model_xgb = joblib.load('xgboost_model_multicurrency.pkl')

# Load multi-currency data
X = np.load('X_lstm_with_indicators_multi.npy')  # (samples, 60, 6)
y = np.load('y_lstm_with_indicators_multi.npy')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# LSTM Predictions
pred_lstm = model_lstm.predict(X_test)

# XGBoost Predictions (requires flattening)
X_xgb_test = X_test.reshape(X_test.shape[0], -1)
pred_xgb = model_xgb.predict(X_xgb_test)

# === Inverse transform utility ===
def inverse_transform(preds_1d, scaler, features):
    padded = np.concatenate([preds_1d.reshape(-1, 1), np.zeros((len(preds_1d), len(features)-1))], axis=1)
    return scaler.inverse_transform(padded)[:, 0]  # Only extract the inverse of 'close'

# For LSTM and XGBoost predictions, always inverse-transform both predictions and y_test before plotting or metrics

# Example for LSTM:
# lstm_preds_real = inverse_transform(lstm_preds, scaler, features)
# y_test_real = inverse_transform(y_test, scaler, features)
# plt.plot(y_test_real, label='Actual')
# plt.plot(lstm_preds_real, label='Predicted')

# Example for XGBoost:
# xgb_preds_real = inverse_transform(xgb_preds, scaler, features)
# y_test_real = inverse_transform(y_test, scaler, features)
# plt.plot(y_test_real, label='Actual')
# plt.plot(xgb_preds_real, label='Predicted')

# Plot predictions
plt.figure(figsize=(10, 5))
y_test_real = inverse_transform(y_test, scaler, features)
pred_lstm_real = inverse_transform(pred_lstm, scaler, features)
pred_xgb_real = inverse_transform(pred_xgb, scaler, features)
plt.plot(y_test_real, label="Actual", color='black')
plt.plot(pred_lstm_real, label="LSTM", linestyle='dashed', alpha=0.8)
plt.plot(pred_xgb_real, label="XGBoost", linestyle='dotted', alpha=0.8)
plt.title("Multi-Currency Model Predictions Comparison (Test Set)")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: print errors
mse_lstm = mean_squared_error(y_test_real, pred_lstm_real)
mse_xgb = mean_squared_error(y_test_real, pred_xgb_real)
print(f"✅ LSTM MSE: {mse_lstm:.5f}")
print(f"✅ XGBoost MSE: {mse_xgb:.5f}")


# In[10]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from lstm_predictor import get_lstm_predictions
from xgb_predictor import get_xgb_predictions
import importlib
import lstm_predictor
importlib.reload(lstm_predictor)

# Load dataset with indicators
df = pd.read_csv("all_currencies_with_indicators.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
features = ['close', 'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal']

# Get LSTM predictions and scaler
lstm_preds, X_lstm, scaler = get_lstm_predictions("lstm_model.keras", df, features)

# Get XGBoost predictions
xgb_preds, X_xgb = get_xgb_predictions("xgboost_model_multicurrency.pkl", df)

# Prepare true values for comparison (aligned with preds)
df_clean = df[features].dropna().reset_index(drop=True)
y_true = df_clean['close'].values[60:]  # match LSTM output length

# === Inverse transform predictions ===
# Step 1: Pad predictions to match scaler input shape
def inverse_transform(preds_1d, scaler, features):
    padded = np.concatenate([preds_1d.reshape(-1, 1), np.zeros((len(preds_1d), len(features)-1))], axis=1)
    return scaler.inverse_transform(padded)[:, 0]  # Only extract the inverse of 'close'

lstm_preds_real = inverse_transform(lstm_preds, scaler, features)
xgb_preds_real = inverse_transform(xgb_preds, scaler, features)

# Now MSE in real price space
mse_lstm = mean_squared_error(y_true, lstm_preds_real)
mse_xgb = mean_squared_error(y_true, xgb_preds_real)

# === Plot ===
plt.figure(figsize=(12, 5))
y_true_real = inverse_transform(y_true, scaler, features)
plt.plot(y_true_real, label='Actual Close', linewidth=1)
plt.plot(lstm_preds_real, label='LSTM Predicted', alpha=0.7)
plt.plot(xgb_preds_real, label='XGBoost Predicted', alpha=0.7)
plt.title("Multi-Currency Forecast: Actual vs Predicted (Real Price)")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print results
print(f"✅ Real LSTM MSE: {mse_lstm:.5f}")
print(f"✅ Real XGBoost MSE: {mse_xgb:.5f}")




# In[ ]:




