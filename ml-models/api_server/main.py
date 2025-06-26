from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Load the HMM model
model_path = os.path.join(os.path.dirname(__file__), "../regime/hmm_model.pkl")
model = joblib.load(model_path)

# Define input schema
class FeatureInput(BaseModel):
    features: list[float]  # Expecting e.g., [log_return, volatility]

@app.get("/")
def health_check():
    return {"status": "ML API is running 🚀"}

@app.post("/predict-regime")
def predict(input: FeatureInput):
    try:
        X = np.array([input.features])
        regime = model.predict(X)
        return {"regime": int(regime[0])}
    except Exception as e:
        return {"error": str(e)}

