import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TWELVE_DATA_API_KEY")
print("✅ Using API Key:", API_KEY)

params = {
    "symbol": "EUR/USD",
    "interval": "4h",
    "outputsize": 100,
    "apikey": API_KEY
}

url = "https://api.twelvedata.com/time_series"
response = requests.get(url, params=params)

data = response.json()

if "values" in data:
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    numeric_cols = ["open", "high", "low", "close"]  # exclude 'volume'
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv("eurusd_4h.csv", index=False)
    print("✅ CSV saved as `eurusd_4h.csv`")
else:
    print("❌ Failed to fetch data:", data.get("message", data))


