# xgb_predictor.py

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

def load_xgb_model(path="xgboost_model_with_indicators.pkl"):
    return joblib.load(path)

def prepare_xgb_input(df, sequence_length=60):
    features = ['close', 'sma_20', 'ema_20', 'rsi', 'macd', 'macd_signal']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])  # Use all 6 features
    X = []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i - sequence_length:i])
    return np.array(X)

def predict_xgb(model, X):
    # Flatten input from (samples, 60, 6) to (samples, 360)
    if len(X.shape) == 3:
        X = X.reshape(X.shape[0], -1)
    return model.predict(X)


def get_xgb_predictions(model_path, df):
    model = load_xgb_model(model_path)
    X = prepare_xgb_input(df)
    preds = predict_xgb(model, X)
    return preds, X



