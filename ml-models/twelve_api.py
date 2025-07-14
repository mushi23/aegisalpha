# twelve_data_fetcher.py

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TWELVE_DATA_API_KEY")

BASE_URL = "https://api.twelvedata.com/time_series"
INTERVAL = "4h"
YEARS = 10
OUTPUTSIZE = 500  # Max per request
SYMBOLS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", 
    "NZD/USD", "USD/CAD", "EUR/JPY", "GBP/JPY", "EUR/GBP"
]

def fetch_historical_data(symbol, interval="4h", years=10):
    now = datetime.utcnow()
    start_date = now - timedelta(days=365 * years)
    dfs = []

    print(f"📈 Fetching {symbol} from {start_date.date()} to {now.date()}...")

    while start_date < now:
        end_date = start_date + timedelta(days=100)  # Pull in ~100-day blocks
        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": API_KEY,
            "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S"),
            "format": "JSON",
            "outputsize": OUTPUTSIZE,
        }

        try:
            response = requests.get(BASE_URL, params=params)
            data = response.json()

            if "values" not in data:
                print(f"❌ Error fetching {symbol}: {data}")
                break

            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime")
            df = df.astype(float, errors="ignore")

            dfs.append(df)

            # Update window
            start_date = df["datetime"].max()
            time.sleep(1)  # avoid hitting rate limits

        except Exception as e:
            print(f"🔥 Exception fetching {symbol}: {e}")
            break

    if dfs:
        full_df = pd.concat(dfs).drop_duplicates("datetime").sort_values("datetime")
        full_df.to_csv(f"data/{symbol.replace('/', '_')}_4h.csv", index=False)
        print(f"✅ Saved: {symbol} with {len(full_df)} rows.")
    else:
        print(f"⚠️ No data fetched for {symbol}.")

def fetch_all_symbols():
    os.makedirs("data", exist_ok=True)
    for symbol in SYMBOLS:
        fetch_historical_data(symbol, interval=INTERVAL, years=YEARS)

if __name__ == "__main__":
    fetch_all_symbols()
