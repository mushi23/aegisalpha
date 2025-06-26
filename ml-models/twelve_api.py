# twelve_api.py

from dotenv import load_dotenv
import os
import requests
import pandas as pd

load_dotenv()
API_KEY = os.getenv("TWELVE_DATA_API_KEY")

def fetch_forex_dataframe(symbol="EUR/USD", interval="30min", output_size=500):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": output_size,
        "apikey": API_KEY,
        "format": "JSON"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "values" not in data:
        raise Exception(f"Error in API response: {data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df = df.astype(float, errors="ignore")
    return df


