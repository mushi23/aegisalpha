# lstm_predictor.py

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def load_lstm_model(path="lstm_model.h5"):
    return load_model(path, compile=False)


def prepare_lstm_input(df, features, sequence_length=60):
    df = df[features].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X = []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i - sequence_length:i])

    return np.array(X), scaler

def predict_lstm(model, X):
    return model.predict(X)

def get_lstm_predictions(model_path, df, features):
    model = load_lstm_model(model_path)
    X, scaler = prepare_lstm_input(df, features)
    preds = predict_lstm(model, X)
    return preds, X, scaler
