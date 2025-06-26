# ml-models/alpha.py

from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_KEY")

def fetch_fx_intraday(from_symbol="EUR", to_symbol="USD", interval="5min"):
    url = "https://www.alphavantage.co/query"
    params = {
    "function": "FX_DAILY",
    "from_symbol": from_symbol,
    "to_symbol": to_symbol,
    "outputsize": "compact",
    "apikey": API_KEY
}


    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Error fetching data:", response.status_code)
        return None

    data = response.json()
    
    if "Time Series FX (" + interval + ")" not in data:
        print("Error in response:", data)
        return None

    return data["Time Series FX (" + interval + ")"]


# For testing:
if __name__ == "__main__":
    data = fetch_fx_intraday()
    print(json.dumps(data, indent=2))
