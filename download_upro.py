import os
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import urllib.request
import json
import time
from core.data_lake import MarketDataLake

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
api_key = os.getenv("ALPACA_API_KEY", "")
secret_key = os.getenv("ALPACA_SECRET_KEY", "")

lake = MarketDataLake()

def download_alpaca_history(sym, tf, start_year=2020):
    tf_map = {"5m": "5Min", "15m": "15Min", "60m": "1Hour"}
    alpaca_tf = tf_map[tf]
    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    
    all_bars = []
    
    for year in range(start_year, 2027):
        start_str = f"{year}-01-01T00:00:00Z"
        end_str = f"{year}-12-31T23:59:59Z"
        page_token = None
        year_bars = []
        while True:
            url = (f"{base_url}?symbols={sym}&timeframe={alpaca_tf}"
                   f"&start={start_str}&end={end_str}"
                   f"&feed=iex&limit=10000")
            if page_token:
                url += f"&page_token={page_token}"
                
            req = urllib.request.Request(url, headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
                "Accept": "application/json"
            })
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    bars = data.get("bars", {}).get(sym, [])
                    year_bars.extend(bars)
                    page_token = data.get("next_page_token")
                    if not page_token:
                        break
            except Exception as e:
                break
        all_bars.extend(year_bars)
            
    if all_bars:
        df = pd.DataFrame(all_bars)
        df.rename(columns={'t': 'datetime', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        # Fix datetime to be compatible
        df['datetime'] = df['datetime'].str.replace('T', ' ').str.replace('Z', '')
        lake.insert_candles(sym, tf, df)

print("Downloading UPRO...")
for tf in ["15m", "60m", "5m"]: download_alpaca_history("UPRO", tf)
print("Downloading SPXU...")
for tf in ["15m", "60m", "5m"]: download_alpaca_history("SPXU", tf)
print("Downloading SPY...")
for tf in ["15m", "60m", "5m"]: download_alpaca_history("SPY", tf)
print("Done.")
