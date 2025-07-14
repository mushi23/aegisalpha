import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

# Load and sort dataset
df = pd.read_csv("all_currencies_with_indicators.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(['pair', 'datetime'])

# FIXED: Use only technical indicators as features, NOT the target variable
features = ['sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal']  # Removed 'close'
target = 'close'

print(f"Features: {features}")
print(f"Target: {target}")

# Group by currency pair to avoid mixing different scales
all_predictions = []
all_actuals = []
all_models = []

for pair in df['pair'].unique():
    print(f"\nProcessing {pair}...")
    
    # Get data for this pair only
    pair_df = df[df['pair'] == pair].copy()
    pair_df = pair_df[features + [target]].dropna().reset_index(drop=True)
    
    if len(pair_df) < 100:  # Skip pairs with insufficient data
        print(f"Skipping {pair} - insufficient data ({len(pair_df)} rows)")
        continue
    
    # Create lagged features (proper time series approach)
    sequence_length = 10  # Reduced from 60 for better performance
    
    X_sequences = []
    y_targets = []
    
    for i in range(sequence_length, len(pair_df)):
        # Use only technical indicators for features
        feature_seq = pair_df.iloc[i-sequence_length:i][features].values
        X_sequences.append(feature_seq.flatten())  # Flatten the sequence
        y_targets.append(pair_df.iloc[i][target])
    
    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)
    
    print(f"{pair} - X shape: {X_sequences.shape}, y shape: {y_targets.shape}")
    
    # Train/test split (time series split)
    split_idx = int(len(X_sequences) * 0.8)
    X_train = X_sequences[:split_idx]
    X_test = X_sequences[split_idx:]
    y_train = y_targets[:split_idx]
    y_test = y_targets[split_idx:]
    
    # Scale features (separately for train/test to avoid leakage)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        early_stopping_rounds=10
    )
    
    # Train with early stopping
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"{pair} - MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    print(f"{pair} - Price range: {y_test.min():.4f} to {y_test.max():.4f}")
    
    # Store results
    all_predictions.extend(y_pred)
    all_actuals.extend(y_test)
    all_models.append((pair, model, scaler))

# Overall evaluation
overall_mse = mean_squared_error(all_actuals, all_predictions)
overall_mae = mean_absolute_error(all_actuals, all_predictions)
overall_rmse = np.sqrt(overall_mse)

print(f"\n=== OVERALL RESULTS ===")
print(f"Total predictions: {len(all_predictions)}")
print(f"Overall MSE: {overall_mse:.6f}")
print(f"Overall MAE: {overall_mae:.6f}")
print(f"Overall RMSE: {overall_rmse:.6f}")

# Plot results
plt.figure(figsize=(15, 10))

# Plot 1: Overall predictions vs actual
plt.subplot(2, 2, 1)
plt.plot(all_actuals[:500], label='Actual', alpha=0.7)
plt.plot(all_predictions[:500], label='Predicted', alpha=0.7)
plt.title('XGBoost Predictions vs Actual (First 500 points)')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Plot 2: Scatter plot
plt.subplot(2, 2, 2)
plt.scatter(all_actuals, all_predictions, alpha=0.5)
plt.plot([min(all_actuals), max(all_actuals)], [min(all_actuals), max(all_actuals)], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual Scatter Plot')
plt.grid(True)

# Plot 3: Residuals
residuals = np.array(all_predictions) - np.array(all_actuals)
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=50, alpha=0.7)
plt.xlabel('Residuals (Predicted - Actual)')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.grid(True)

# Plot 4: Residuals over time
plt.subplot(2, 2, 4)
plt.plot(residuals[:500])
plt.xlabel('Time Steps')
plt.ylabel('Residuals')
plt.title('Residuals Over Time (First 500 points)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Save the best model (you can modify this to save all models)
if all_models:
    best_pair, best_model, best_scaler = all_models[0]  # Save first model as example
    joblib.dump(best_model, f"xgboost_model_{best_pair}_fixed.pkl")
    joblib.dump(best_scaler, f"xgboost_scaler_{best_pair}_fixed.pkl")
    print(f"\nSaved model for {best_pair}")

print("\n✅ Fixed XGBoost training completed!") 